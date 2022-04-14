from cv2 import norm
import torch
from mmdet.apis import init_detector, inference_detector
from mmcv import Config, DictAction
from mmdet.datasets import replace_ImageToTensor
from mmdet.datasets.pipelines import Compose
from mmcv.ops import RoIPool
import warnings
from mmcv.parallel import collate, scatter
import numpy as np
from functools import partial

from mmdet.core.export import build_model_from_cfg, preprocess_example_input


def parse_normalize_cfg(test_pipeline):
    transforms = None
    for pipeline in test_pipeline:
        if 'transforms' in pipeline:
            transforms = pipeline['transforms']
            break
    assert transforms is not None, 'Failed to find `transforms`'
    norm_config_li = [_ for _ in transforms if _['type'] == 'Normalize']
    assert len(norm_config_li) == 1, '`norm_config` should only have one'
    norm_config = norm_config_li[0]
    return norm_config


def pytorch2trace(model,
    input_img,
    input_shape,
    normalize_cfg,
    output_file='tmp.pt',
    verify=False,
    test_img=None,
    show=False,
    skip_postprocess=False):

    input_config = {
        'input_shape': input_shape,
        'input_path': input_img,
        'normalize_cfg': normalize_cfg
    }


    one_img, one_meta = preprocess_example_input(input_config)
    # print('-----------one-----------')
    # print(one_img, one_meta)
    img_list, img_meta_list = [one_img], [[one_meta]]

    if skip_postprocess:
        warnings.warn('Not all models support export onnx without post '
                      'process, especially two stage detectors!')
        model.forward = model.forward_dummy
        script_model = torch.jit.trace(model, one_img)
        script_model.save(output_file)
        print(f'Successfully exported ONNX model without '
              f'post process: {output_file}')
        return

    # replace original forward function
    origin_forward = model.forward
    # model.forward = partial(
    #     model.forward,
    #     img_metas=img_meta_list,
    #     return_loss=False,
    #     rescale=False)

    model.forward = model.forward_test

    # script_model = torch.jit.trace(model, [one_img])
    script_model = torch.jit.script(model)
    script_model.save(output_file)

    model.forward = origin_forward

    if verify:
        # check by onnx
        jit_model = torch.jit.load(output_file)
        # onnx.checker.check_model(onnx_model)

        if test_img is None:
            input_config['input_path'] = input_img

        # prepare input once again
        one_img, one_meta = preprocess_example_input(input_config)
        img_list, img_meta_list = [one_img], [[one_meta]]

        # get pytorch output
        with torch.no_grad():
            pytorch_results = model(
                img_list,
                img_metas=img_meta_list,
                return_loss=False,
                rescale=True)[0]

        img_list = [_.cuda().contiguous() for _ in img_list]
        # get ort output
        jit_results = jit_model(
            img_list, img_metas=img_meta_list, return_loss=False)[0]
        # visualize predictions

        score_thr = 0.3

        if show:
            out_file_jt, out_file_pt = None, None
        else:
            out_file_jt, out_file_pt = 'show-ort.png', 'show-pt.png'

        show_img = one_meta['show_img']

        model.show_result(
            show_img,
            pytorch_results,
            score_thr=score_thr,
            show=True,
            win_name='PyTorch',
            out_file=out_file_pt)
        jit_model.show_result(
            show_img,
            jit_results,
            score_thr=score_thr,
            show=True,
            win_name='Jit Trace',
            out_file=out_file_jt)


if __name__ == '__main__':

    config_file = 'configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'

    checkpoint_file = '/root/checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'

    input_img = 'demo/demo.jpg'

    cfg = Config.fromfile(config_file)

    # print('cfg: \n', cfg)
    img_scale = cfg.test_pipeline[1]['img_scale']
    print('-------------image scale----------------\n', img_scale)
    input_shape = (1, 3, 400, 600)

    # input_shape = (1, 3, img_scale[1], img_scale[0])

    model = build_model_from_cfg(config_file, checkpoint_file)

    normalize_cfg = parse_normalize_cfg(cfg.test_pipeline)

    pytorch2trace(
        model=model,
        input_img=input_img,
        input_shape=input_shape,
        normalize_cfg=normalize_cfg,
        output_file='faster_rcnn.pt'
    )

# model = init_detector(config_file, checkpoint_file, device='cpu')


# inference_detector(model, 'demo/demo.jpg')

# print('-------------model---------------')
# print(model)

# print('-------------model module--------------')
# for m in model.modules():
#     print(type(m))

# imgs = mmcv.imread('demo/demo.jpg')

# ori_imgs = imgs
# if isinstance(imgs, (list, tuple)):
#     is_batch = True
# else:
#     imgs = [imgs]
#     is_batch = False

# cfg = model.cfg
# device = next(model.parameters()).device  # model device

# if isinstance(imgs[0], np.ndarray):
#     cfg = cfg.copy()
#     # set loading pipeline type
#     cfg.data.test.pipeline[0].type = 'LoadImageFromWebcam'

# cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
# test_pipeline = Compose(cfg.data.test.pipeline)

# datas = []
# for img in imgs:
#     # prepare data
#     if isinstance(img, np.ndarray):
#         # directly add img
#         data = dict(img=img)
#     else:
#         # add information into dict
#         data = dict(img_info=dict(filename=img), img_prefix=None)
#         # build the data pipeline
#     data = test_pipeline(data)
#     datas.append(data)

# data = collate(datas, samples_per_gpu=len(imgs))
# # just get the actual data from DataContainer
# data['img_metas'] = [img_metas.data[0] for img_metas in data['img_metas']]
# data['img'] = [img.data[0] for img in data['img']]
# if next(model.parameters()).is_cuda:
#     # scatter to specified GPU
#     data = scatter(data, [device])[0]
# else:
#     for m in model.modules():
#         assert not isinstance(
#             m, RoIPool
#         ), 'CPU inference with RoIPool is not supported currently.'

# # forward the model
# with torch.no_grad():
#     results = model(return_loss=False, rescale=True, **data)


# print(data['img_metas'][0][0])
# print(type(data['img'][0]))

# # if not is_batch:
# #     print(results[0])
# # else:
# #     print(results)

# # input_tensor = torch.rand(1,3,224,224

# # print('---------data imge metas---------')
# # print(data['img_metas'])
# # print('---------data imge metas---0---------')
# # print(data['img_metas'][0])