from datasets.shapenet_pointflow import get_datasets
import numpy as np
import torch
from metrics.evaluation_metrics import compute_all_metrics
from pprint import pprint
from tqdm import tqdm
import os
from metrics.evaluation_metrics import jsd_between_point_cloud_sets as JSD


def get_test_dataset(path, cates = ['chair']):
    # using the same parameters as point flow
    class Args: pass
    args = Args()
    args.data_dir = path
    args.dataset_type = 'shapenet15k'
    args.tr_max_sample_points = 2048
    args.te_max_sample_points = 2048
    args.dataset_scale = 1.
    args.normalize_per_shape = False
    args.normalize_std_per_axis = False
    args.cates = cates

    _, test_dataset = get_datasets(args)

    return test_dataset

def get_test_loader(path, cates = ['chair']):
    test_dataset = get_test_dataset(path, cates)

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
    )

    return test_loader


def evaluate_gen(path, model, sampler, save_path='./results/', cates = ['chair']):

    # try to load generated files
    try:
        sample_pcs = torch.tensor(np.load(os.path.join(save_path, 'generated_pcs.npy')))
        ref_pcs = torch.tensor(np.load(os.path.join(save_path, 'reference_pcs.npy')))
    
    except:
        print('Generating new data...')

        loader = get_test_loader(path, cates)

        all_sample = []
        all_ref = []

        for data in tqdm(loader):
            idx_b, te_pc = data['idx'], data['test_points']
            te_pc = te_pc.cuda()

            # number of samples, number of points
            B, N = te_pc.shape[0], te_pc.shape[1]
            
            out_pc = sampler.sample(model, B, n_points=N) #sampler.sample(model, (B, N, 3))[-1] #model.sample(B, N)
            out_pc = out_pc.cuda()
            

            # denormalize 
            m, s = data['mean'].float(), data['std'].float()
            m = m.cuda()
            s = s.cuda()
            out_pc = out_pc * s + m
            te_pc = te_pc * s + m

            all_sample.append(out_pc)
            all_ref.append(te_pc)


        sample_pcs = torch.cat(all_sample, dim=0)
        ref_pcs = torch.cat(all_ref, dim=0)
        

        # save results to a file
        np.save(os.path.join(save_path, 'generated_pcs.npy'), sample_pcs.cpu().numpy())
        np.save(os.path.join(save_path, 'reference_pcs.npy'), ref_pcs.cpu().numpy())

    # -------------------------------------------------------------------------------------
    # # per sample centering
    # sample_pcs = sample_pcs - sample_pcs.mean(dim=1, keepdim=True)
    # ref_pcs = ref_pcs - ref_pcs.mean(dim=1, keepdim=True)
    # # per sample scaling
    # dist = (sample_pcs * sample_pcs).sum(dim=-1, keepdim=True).sqrt().max(dim=1).values
    # sample_pcs = sample_pcs / dist.unsqueeze(-1)
    # dist = (ref_pcs * ref_pcs).sum(dim=-1, keepdim=True).sqrt().max(dim=1).values
    # ref_pcs = ref_pcs / dist.unsqueeze(-1)
    # -------------------------------------------------------------------------------------

    print((sample_pcs * sample_pcs).sum(dim=-1, keepdim=True).sqrt().max(), (ref_pcs * ref_pcs).sum(dim=-1, keepdim=True).sqrt().max())
    print(sample_pcs.mean(), ref_pcs.mean())

    print(f'Comparing {sample_pcs.shape[0]} generated samples of shape {list(sample_pcs.shape[1:])} to {ref_pcs.shape[0]} original samples of shape {list(ref_pcs.shape[1:])}')
    results = compute_all_metrics(sample_pcs, ref_pcs, batch_size=32)
    results = {k: (v.cpu().detach().item()
                   if not isinstance(v, float) else v) for k, v in results.items()}
    pprint(results)

    jsd = JSD(sample_pcs.numpy(), ref_pcs.numpy())
    pprint('JSD: {}'.format(jsd))


def main():
    #path = "/home/tourloid/Desktop/PhD/Data/ShapeNetCore.v2.PC15k"
    path = "/home/vvrbeast/Desktop/Giannis/Data/ShapeNetCore.v2.PC15k"

    from models.ddpm_unet import SPVUnet
    from utils.schedulers import DDPMSparseSchedulerGPU

    model = SPVUnet(voxel_size=0.1, nfs=(32, 64, 128, 256), num_layers=1, pres=1e-5)

    checkpoint_path = './checkpoints/spvcnn_0.1_32_64_128_256_constant_lr_1900.pt'

    checkpoint = torch.load(checkpoint_path)['state_dict']
    model.load_state_dict(checkpoint)
    model.cuda().eval()
    
    ddpm_sched = DDPMSparseSchedulerGPU(n_steps=1000, beta_min=0.0001, beta_max=0.02)

    evaluate_gen(path, model, ddpm_sched, save_path='./results/')


if __name__ == "__main__":
    main()

