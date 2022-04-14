# faster rcnn
* 详细代码见1.py

  这里面`checkpoint_file`需要下载：
  ```
  # download the checkpoint from model zoo and put it in `checkpoints/`
  # url: https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth
  ```

  其实只有`61-75`行的代码在发挥作用：
  ```python
  # replace original forward function
      origin_forward = model.forward
      # model.forward = partial(
      #     model.forward,
      #     img_metas=img_meta_list,
      #     return_loss=False,
      #     rescale=False)

      # --- 因为推理的时候走的是`forward_test`，我让`model.forward = model.forward_test`了.---
      model.forward = model.forward_test

      # script_model = torch.jit.trace(model, [one_img])
      script_model = torch.jit.script(model)
      script_model.save(output_file)

      model.forward = origin_forward
  ```
## 修改的文件
### mmdetection/mmdet/models/detectors/base.py
因为112行`forward_test(self, imgs, img_metas, **kwargs)`里的`**kwargs`报错，我看推理时也没用上，就把相关代码删去。
### mmdetection/mmdet/models/backbones/resnet.py
* 275行`def _inner_forward(x):`会报错，我看304行的`if`判断语句中`self.with_cp`为`False`，就直接删去`_inner_forward`，将里面的代码提出来放到`forward`里，
* 紧接着是`getattr`报错：
  ```
  RuntimeError: 
  getattr's second argument must be a string literal:
    File "/root/mmdetection/mmdet/models/backbones/resnet.py", line 262
      def norm1(self):
          """nn.Module: normalization layer after the first convolution layer"""
          return getattr(self, self.norm1_name)
                ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ <--- HERE
  ```
  这里`self.norm1_name="bn1"`，把相应的`norm1_name`换成`"bn1"`后，报了以下错:
  ```
  RuntimeError: 
  '__torch__.torch.nn.modules.batchnorm.BatchNorm2d (of Python compilation unit at: 0x560daa4db980)' object has no attribute or method '__call__'. Did you forget to initialize an attribute in __init__()?:
    File "/root/mmdetection/mmdet/models/backbones/resnet.py", line 281
          identity = x
          out = self.conv1(x)
          out = self.norm1(out)
                ~~~~~~~~~~ <--- HERE
          out = self.relu(out)
  ```
  目前在弄这个bug。


