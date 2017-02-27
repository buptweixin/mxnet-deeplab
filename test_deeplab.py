#!/usr/bin/env python
# encoding: utf-8

def getpallete(num_cls):
    Building = [0, 0, 255]
    Tree = [0, 255, 0]
    Clutter = [255, 0, 0]
    LowVegetation = [0, 255, 255]
    Car = [255, 255, 0]
    ImpervousSurface = [255, 255, 255]
    labelsColors = [Building, Tree, Clutter, LowVegetation, Car, ImpervousSurface]

    n = num_cls
    pallete = [0]*(n*3)
    for j in xrange(0, n):
        lab = j
        pallete[j*3+0] = labelsColors[j][0]
        pallete[j*3+1] = labelsColors[j][1]
        pallete[j*3+2] = labelsColors[j][2]
    return pallete

pallete = getpallete(6)


def get_data(img_path):
    mean = np.array([123.68, 116.779, 103.939])
    img = Image.open(img_path)
    img = np.array(img, dtype=np.float32)
    reshaped_mean = mean.reshape(1, 1, 3)
    img = img - reshaped_mean
    img = np.swapaxes(img, 0, 2)
    img = np.swapaxes(img, 1, 2)
    img = np.expand_dims(img, axis=0)
    return img

model_prefix = "DeepLab-V2"
epoch = 50
img = "./ISPRS/test_cropped_513_513/top_mosaic_09cm_area300.tif"
seg = "result.png"

def main():

    deeplab_symbol, deeplab_args, deeplab_aux = \
    mx.model.load_checkpoint(model_prefix, epoch)
    deeplab_args['data'] = mx.nd.array(get_data(img), ctx)
    data_shape = deeplab_args['data'].shape
    label_shape = (1, 65*65)
    deeplab_args['softmax_label'] = mx.nd.empty(label_shape, ctx)

    executor = deeplab_symbol.bind(ctx, deeplab_args, args_grad=None,
                                  grad_req='null', aux_states=deeplab_aux)
    executor.forward(is_train=False)

    output = executor.outputs[0]
    out_img = np.uint8(np.squeeze(output.asnumpy().argmax(axis=1)))
    out_img = cv2.resize(out_img, (513, 513))
    out_img = Image.fromarray(out_img)
    out_img.putpalette(pallete)
    out_img.save(seg)



main()
