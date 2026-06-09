import time
import os
import torch
import torch.nn.functional as F
import numpy as np
from utils import dataloader
from tqdm import tqdm
from utils.Metrics import Evaluator
from PIL import Image
from models.baseS2FM import STMbaseS2FM
from models.baseline import baseline
from models.FullNet import FullNet
from models.baseCAAS import baseCAAS
from models.mcd import mcd
import torchvision.transforms as transforms

start_time = time.time()


def test(test_loader, Eva_test, save_path, net, testsize):
    print("Validation process started (evaluating each tile directly).")

    net.train(False)
    net.eval()

    for i, (A, B, mask, base_names, infos) in enumerate(tqdm(test_loader, desc="Testing")):
        with torch.no_grad():
            A = A.cuda()
            B = B.cuda()
            mask = mask.cuda() 
            outputs = net(A, B)
            pred_tile = F.sigmoid(outputs[0])
            pred_for_eval = (pred_tile >= 0.5).to(torch.int64)
            Eva_test.add_batch(mask.squeeze(1).cpu(), pred_for_eval.squeeze(1).cpu())
            for j in range(pred_tile.shape[0]):
                base_name = base_names[j]
                info = {key: val[j].item() for key, val in infos.items()}
                row, col = info['row'], info['col']
                tile_save_name = f"{os.path.splitext(base_name)[0]}_tile_{row}_{col}.png"
                final_save_path = os.path.join(save_path, tile_save_name)
                bin_mask = (pred_tile[j].squeeze() >= 0.5).cpu().numpy()  # 0/1 bool
                bin_uint8 = (bin_mask * 255).astype(np.uint8)  # 0/255 uint8
                im = Image.fromarray(bin_uint8)
                im.save(final_save_path)
    print("\nAll test tiles processed. Calculating final metrics...")
    IoU = Eva_test.Intersection_over_Union()
    Pre = Eva_test.Precision()
    Recall = Eva_test.Recall()
    F1 = Eva_test.F1()
    OA = Eva_test.OA()
    Kappa = Eva_test.Kappa()

    print('[Test] F1: %.4f, Precision:%.4f, Recall: %.4f, OA: %.4f, Kappa: %.4f, IoU: %.4f' % (F1[1], Pre[1], Recall[1],
                                                                                               OA[1], Kappa, IoU[1]))
    print('F1-Score: Precision: Recall: OA: Kappa: IoU: ')
    print(
        '{:.2f}\\{:.2f}\\{:.2f}\\{:.2f}\\{:.2f}\\{:.2f}'.format(F1[1] * 100, Pre[1] * 100, Recall[1] * 100, OA[1] * 100,
                                                                Kappa * 100, IoU[1] * 100))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchsize', type=int, default=8, help='testing batch size')
    parser.add_argument('--testsize', type=int, default=256, help='testing image size (tile size)')
    parser.add_argument('--gpu_id', type=str, default='0', help='train use gpu')
    parser.add_argument('--data_name', type=str, default='WHU', help='the test dataset name')
    parser.add_argument('--model_name', type=str, default='mcd', help='the model name')
    parser.add_argument('--save_path', type=str, default='/root/autodl-tmp/data/WHU/WHU/results_mcd')
    parser.add_argument('--model_load_path', type=str,
                        default='/root/autodl-tmp/data/WHU/WHU/mcd/_best_iou.pth',
                        help='path to the trained model weight')

    opt = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
    print(f'Using GPU {opt.gpu_id}')
    if opt.data_name == 'LEVIR':
        opt.test_root = '/root/autodl-tmp/data/LEVIR-CD/test/'
    elif opt.data_name == 'SYSU':
        opt.test_root = "/root/autodl-tmp/data/SYSU/test/"
    elif opt.data_name == 'WHU':
        opt.test_root = '/root/autodl-tmp/data/WHU/test/'
    elif opt.data_name == 'CDD':
        opt.test_root = '/root/autodl-tmp/data/CDD/Real/test'
    elif opt.data_name == 'CLCD':
        opt.test_root = '/root/autodl-tmp/data/CLCD/test'

    test_loader = dataloader.get_test_loader(
        opt.test_root,
        opt.batchsize,
        opt.testsize,
        num_workers=8,
        shuffle=False,
        pin_memory=True
    )

    Eva_test = Evaluator(num_class=2)

    model = None

    if opt.model_name == 'FullNet':
        model= FullNet(img_size=[256, 256], in_channels=3, num_classes=1,
                        dims=[96, 192, 384, 768], depths=[2, 2, 8, 2], 
                        deep_supervision=True,  
                        s2fm_r1=8, s2fm_r2=8, ssm_d_state=16,
                        alpha_inits=[0,0,0,0]).cuda()
    elif opt.model_name == 'mcd':
        model=mcd(
            img_size=[256,256],  
            num_classes=1,
            in_channels=3,
            dims=[96, 192, 384, 768],  
            depths=[2, 2, 8, 2],  
            deep_supervision=True,
        ).cuda()
    else:
        raise ValueError(f"Model {opt.model_name} is not recognized.")

    if opt.model_load_path is not None and os.path.isfile(opt.model_load_path):
        print(f"Loading model from {opt.model_load_path}")
        checkpoint = torch.load(opt.model_load_path, map_location=f'cuda:{opt.gpu_id}')
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Model loaded successfully!")
    else:
        raise FileNotFoundError(f"No valid model file found at {opt.model_load_path}")

    if not os.path.exists(opt.save_path):
        os.makedirs(opt.save_path)
        print(f"Created save directory: {opt.save_path}")

    test(test_loader, Eva_test, opt.save_path, model, opt.testsize)

    end_time = time.time()
    print(f'Total testing time: {end_time - start_time:.2f} seconds')

