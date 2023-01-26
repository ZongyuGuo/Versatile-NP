import time

import torch
import torchvision
import numpy as np
import einops
import wandb
from tqdm import tqdm

import utils
from .base_trainer import BaseTrainer
from trainers import register
from utils import poses_to_rays, get_coord, volume_rendering, batched_volume_rendering


@register('nvs_trainer')
class NvsTrainer(BaseTrainer):

    def make_datasets(self):
        super().make_datasets()

        def get_vislist(dataset, n_vis=8):
            ids = torch.arange(n_vis) * (len(dataset) // n_vis)
            return [dataset[i] for i in ids]

        if hasattr(self, 'train_loader'):
            np.random.seed(0)
            self.vislist_train = get_vislist(self.train_loader.dataset)
        if hasattr(self, 'test_loader'):
            np.random.seed(0)
            self.vislist_test = get_vislist(self.test_loader.dataset)

    def adjust_learning_rate(self):
        base_lr = self.cfg['optimizer']['args']['lr']
        if self.epoch <= round(self.cfg['max_epoch'] * 0.8):
            lr = base_lr
        else:
            lr = base_lr * 0.1
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        self.log_temp_scalar('lr', lr)

    def _adaptive_sample_rays(self, rays_o, rays_d, gt, n_sample):
        B = rays_o.shape[0]
        inds = []
        fg_n_sample = n_sample // 2
        for i in range(B):
            fg = ((gt[i].min(dim=-1).values < 1).nonzero().view(-1)).cpu().numpy()
            if fg_n_sample <= len(fg):
                fg = np.random.choice(fg, fg_n_sample, replace=False)
            else:
                fg = np.concatenate([fg, np.random.choice(fg, fg_n_sample - len(fg), replace=True)], axis=0)
            rd = np.random.choice(rays_o.shape[1], n_sample - fg_n_sample, replace=False)
            inds.append(np.concatenate([fg, rd], axis=0))

        def subselect(x, inds):
            t = torch.empty(B, len(inds[0]), 3, dtype=x.dtype, device=x.device)
            for i in range(B):
                t[i] = x[i][inds[i], :]
            return t

        return subselect(rays_o, inds), subselect(rays_d, inds), subselect(gt, inds)

    def _iter_step(self, data, is_train):
        data = {k: v.cuda() for k, v in data.items()}
        query_imgs = data.pop('query_imgs')

        B, _, _, H, W = query_imgs.shape
        rays_o, rays_d = poses_to_rays(data['query_poses'], H, W, data['query_focals'])

        gt = einops.rearrange(query_imgs, 'b n c h w -> b (n h w) c')
        rays_o = einops.rearrange(rays_o, 'b n h w c -> b (n h w) c')
        rays_d = einops.rearrange(rays_d, 'b n h w c -> b (n h w) c')

        n_sample = self.cfg['train_n_rays']
        if is_train and self.epoch <= self.cfg.get('adaptive_sample_epoch', 0):
            rays_o, rays_d, gt = self._adaptive_sample_rays(rays_o, rays_d, gt, n_sample)
        else:
            ray_ids = np.random.choice(rays_o.shape[1], n_sample, replace=False)
            rays_o, rays_d, gt = map((lambda _: _[:, ray_ids, :]), [rays_o, rays_d, gt])

        # x_tgt is the world coordinate of target points
        x_tgt, z_vals = get_coord(
            rays_o, rays_d, 
            near=data['near'][0],
            far=data['far'][0],
            points_per_ray=self.cfg['train_points_per_ray'],
            rand=is_train,
        )

        pred, loss_kl = self.model_ddp(data, x_tgt, rays_o, z_vals, gt, is_train=is_train)

        loss_mse = ((pred - gt) ** 2).view(B, -1).mean(dim=-1)
        loss_kl = loss_kl.view(B, -1).mean()

        # annealing beta coefficient
        if self.cfg.get('resume_model') is not None:
            beta = self.cfg['Lambda']
        else:
            beta = self.cfg['Lambda'] * min(1.0, self.epoch / 50)

        loss = loss_mse.mean() + loss_kl * beta
        psnr = (-10 * torch.log10(loss_mse)).mean()

        if is_train:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return {'loss': loss.item(), 'psnr': psnr.item(), 'loss_kl': loss_kl.item()}

    def train_step(self, data):
        return self._iter_step(data, is_train=True)

    def evaluate_step(self, data):
        with torch.no_grad():
            return self._iter_step(data, is_train=False)


@register('nvs_evaluator')
class NvsEvaluator(NvsTrainer):

    def _test_time_optimization(self, hyponet, data, n_iters, n_rays=1024):
        B, _, _, H, W = data['support_imgs'].shape
        rays_o, rays_d = poses_to_rays(data['support_poses'], H, W, data['support_focals'])
        
        gt = einops.rearrange(data['support_imgs'], 'b n c h w -> b (n h w) c')
        rays_o = einops.rearrange(rays_o, 'b n h w c -> b (n h w) c')
        rays_d = einops.rearrange(rays_d, 'b n h w c -> b (n h w) c')

        p_lst = []
        for k, v in hyponet.params.items():
            p = torch.nn.Parameter(v.detach())
            p_lst.append(p)
            hyponet.params[k] = p
        
        optimizer = torch.optim.Adam(p_lst, lr=1e-4)

        for i_iter in range(n_iters):
            ray_ids = np.random.choice(rays_o.shape[1], n_rays)
            rays_o_, rays_d_, gt_ = map((lambda _: _[:, ray_ids, :]), [rays_o, rays_d, gt])
            pred_ = volume_rendering(
                hyponet, rays_o_, rays_d_,
                near=data['near'][0],
                far=data['far'][0],
                points_per_ray=self.cfg['train_points_per_ray'],
                use_viewdirs=hyponet.use_viewdirs,
                rand=False,
            )
            mses = ((pred_ - gt_)**2).view(B, -1).mean(dim=-1)
            optimizer.zero_grad()
            mses.sum().backward() # sum, not mean, as every hyponet is independent, though adam doesn't care
            optimizer.step()
            pred_ = mses = None
        
        for k, p in hyponet.params.items():
            hyponet.params[k] = p.data

    def _iter_step(self, data, is_train, step=0):
        assert not is_train
        data = {k: v.cuda() for k, v in data.items()}

        query_imgs = data.pop('query_imgs')
        B, N, _, H, W = query_imgs.shape

        with torch.no_grad():
            B, _, _, H, W = query_imgs.shape
            rays_o, rays_d = poses_to_rays(data['query_poses'], H, W, data['query_focals'])
            
            gt = einops.rearrange(query_imgs, 'b n c h w -> b (n h w) c')
            rays_o = einops.rearrange(rays_o, 'b n h w c -> b (n h w) c')
            rays_d = einops.rearrange(rays_d, 'b n h w c -> b (n h w) c')
            # print(rays_o.shape, rays_d.shape)
            n_sample = self.cfg['render_ray_batch']
            pred = []
            for i in range(0, rays_o.shape[1], n_sample):
                rays_o_, rays_d_, gt_ = map((lambda _: _[:, i: i + n_sample, :]), [rays_o, rays_d, gt])
                x_tgt, z_vals = get_coord(
                    rays_o_, rays_d_, 
                    near=data['near'][0],
                    far=data['far'][0],
                    points_per_ray=self.cfg['train_points_per_ray'],
                    rand=is_train,
                )
                pred_, loss_kl = self.model_ddp(data, x_tgt, rays_o_, z_vals, gt_, is_train=False)
                pred.append(pred_)


        pred = torch.cat(pred, dim=1)
        pred = torch.clamp(pred, min=0.0, max=1.0)
        gt = torch.clamp(gt, min=0.0, max=1.0)
        # print(gt.shape, pred.shape)

        ref = data['support_imgs'][:,0].permute(0, 2, 3, 1)
        # save_img = torch.cat([ref.view(B, 128, 128, 3), pred.view(B, 128, 128, 3), gt.view(B, 128, 128, 3)], dim=2).view(-1, 384, 3)
        # imsave('save_imgs/save_img' + str(step) + '.png', np.squeeze(save_img.cpu().numpy() * 255).astype(np.uint8))
        
        loss_kl = loss_kl.view(B, -1).mean(-1)
        mses = ((pred - gt)**2).view(B * N, -1).mean(dim=-1)
        psnr = (-10 * torch.log10(mses)).mean()
        # print(psnr)

        return {'psnr': psnr.item(), 'loss_kl': loss_kl.mean().item()}
    
    def evaluate_epoch(self):
        self.model_ddp.eval()
        ave_scalars = dict()

        pbar = self.test_loader
        if self.is_master:
            pbar = tqdm(pbar, desc='eval', leave=False)

        t1 = time.time()
        for data in pbar:
            t0 = time.time()
            self.t_data += t0 - t1
            ret = self.evaluate_step(data)
            self.t_model += time.time() - t0

            B = len(next(iter(data.values())))
            for k, v in ret.items():
                if ave_scalars.get(k) is None:
                    ave_scalars[k] = utils.Averager()
                ave_scalars[k].add(v, n=B)

            # --The only thing added-- #
            if self.is_master:
                pbar.set_description(desc=f'eval: psnr={ave_scalars["psnr"].item():.2f}')
            # ------------------------ #
            t1 = time.time()

        if self.distributed:
            self.sync_ave_scalars_(ave_scalars)

        logtext = 'eval:'
        for k, v in ave_scalars.items():
            logtext += f' {k}={v.item():.4f}'
            self.log_temp_scalar('test/' + k, v.item())
        self.log_buffer.append(logtext)