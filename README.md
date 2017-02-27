本项目将Caffe版本的DeepLab v2-LargeFOV网络移植到了MXNet框架.
## 使用方法：
### 依赖
1. mxnet
    version > 0.7
2. opencv及其python接口
### 数据集准备
从百度云下载预训练好的VGG16模型和参数文件到工程根目录:
```shell
# 下载VGG_FC_ILSVRC_16_layers-symbol.json
wget --refer "http://pan.baidu.com/s/1bgz4PC" -O VGG_FC_ILSVRC_16_layers-symbol.json "https://3grauymt1go3nhcttfa3ug.ourdvsss.com/d1.baidupcs.com/file/3990a272c33b0242f02420c9a130d640?bkt=p3-14003990a272c33b0242f02420c9a130d64079c68c30000000003b6a&xcode=2bb72c1609f809d1916aec7057a1518716d653c658a55f46a7103330c9091c9b&fid=1108131987-250528-1034373700851642&time=1488202954&sign=FDTAXGERLBH-DCb740ccc5511e5e8fedcff06b081203-3tYh9R%2F%2F7p9F3te2s1SPKRrY5%2B0%3D&to=sf&fm=Yan,B,U,nc&sta_dx=15210&sta_cs=350&sta_ft=json&sta_ct=7&sta_mt=7&fm2=Yangquan,B,U,nc&newver=1&newfm=1&secfm=1&flow_ver=3&pkey=14003990a272c33b0242f02420c9a130d64079c68c30000000003b6a&sl=72286287&expires=8h&rt=sh&r=968934640&mlogid=1342970328391683748&vuk=3289204393&vbdid=3039681765&fin=VGG_FC_ILSVRC_16_layers-symbol.json&fn=VGG_FC_ILSVRC_16_layers-symbol.json&slt=pm&uta=0&rtype=1&iv=0&isw=0&dp-logid=1342970328391683748&dp-callid=0.1.1&hps=1&csl=400&csign=NuOTkGygYoSPLqoaZF1HHMNTTIA%3D&by=flowserver&wshc_tag=0&wsts_tag=58b42ccb&wsid_tag=3d94f4ae&wsiphost=ipdbm"
# 下载VGG_FC_ILSVR_16_layers-0074.params
wget --refer "http://pan.baidu.com/s/1bgz4PC" -O VGG_FC_ILSVR_16_layers-0074.params "https://qdcache00.baidupcs.com/file/4d805929e82225892ecbee68c33cc648?bkt=p3-0000d1495a68d33493685f8e663ddb61eb06&xcode=2708ca2c3104a7a3eb1c30a0cadc72b6c34f7fb60676eabd1682cb8519c2059f&fid=1108131987-250528-409289218452971&time=1488203032&sign=FDTAXGERLBH-DCb740ccc5511e5e8fedcff06b081203-gJocdHKzbBMjzJQE%2FenZMlZ%2BF%2BU%3D&to=qd00&fm=Nan,B,U,nc&sta_dx=553431816&sta_cs=41&sta_ft=params&sta_ct=7&sta_mt=7&fm2=Nanjing02,B,U,nc&newver=1&newfm=1&secfm=1&flow_ver=3&pkey=0000d1495a68d33493685f8e663ddb61eb06&sl=70123598&expires=8h&rt=sh&r=344308576&mlogid=1342991402114828163&vuk=3289204393&vbdid=3039681765&fin=VGG_FC_ILSVRC_16_layers-0074.params&fn=VGG_FC_ILSVRC_16_layers-0074.params&slt=pm&uta=0&rtype=1&iv=0&isw=0&dp-logid=1342991402114828163&dp-callid=0.1.1&hps=1&csl=241&csign=Zi9ouGXjhj3biOF08bDZFKVcEI8%3D&by=flowserver"
```
下载ISPRS数据集(ss)[https://pan.baidu.com/s/1kUQxs8J]
```shell
wget --refer "https://pan.baidu.com/s/1kUQxs8J" -O ISPRS.tar.gz "https://nj02all01.baidupcs.com/file/e4c13a020304544513d29e545b18c688?bkt=p3-00001ad5dace852e8ae5fb4531387e8864df&fid=3289204393-250528-459728222016&time=1488204092&sign=FDTAXGERLBH-DCb740ccc5511e5e8fedcff06b081203-epsiDCjUMkrcStwOefxj3RLQDbo%3D&to=nj2hb&fm=Yan,B,U,nc&sta_dx=467793561&sta_cs=&sta_ft=gz&sta_ct=0&sta_mt=0&fm2=Yangquan,B,U,nc&newver=1&newfm=1&secfm=1&flow_ver=3&pkey=00001ad5dace852e8ae5fb4531387e8864df&sl=76480590&expires=8h&rt=sh&r=430772044&mlogid=1343275688902104264&vuk=3289204393&vbdid=3039681765&fin=ISPRS.tar.gz&fn=ISPRS.tar.gz&slt=pm&uta=0&rtype=1&iv=0&isw=0&dp-logid=1343275688902104264&dp-callid=0.1.1&hps=1&csl=80&csign=27%2F%2BmaOjbfOdfUYOWleCxsS4EqM%3D&by=flowserver"
# 解压数据集到工程根目录
tar -xzvf ISPRS.tar.gz
```
在`run_deeplab.sh`中设置数据集根路径，比如"root_dir=./ISPRS/"

### 运行
```shell
./run_deeplab.sh
```
部分日志
```shell
INFO:root:Start training with gpu(0)
INFO:root:Epoch[0] Batch [10]   Speed: 5.68 samples/sec Train-accuracy=0.781917
INFO:root:Epoch[0] Batch [20]   Speed: 5.17 samples/sec Train-accuracy=0.789799
INFO:root:Epoch[0] Batch [30]   Speed: 5.17 samples/sec Train-accuracy=0.772426
INFO:root:Epoch[0] Batch [40]   Speed: 5.17 samples/sec Train-accuracy=0.806343
INFO:root:Epoch[0] Batch [50]   Speed: 5.18 samples/sec Train-accuracy=0.847787
INFO:root:Epoch[0] Batch [60]   Speed: 5.18 samples/sec Train-accuracy=0.827195
INFO:root:Epoch[0] Batch [70]   Speed: 5.19 samples/sec Train-accuracy=0.809657
...
...
INFO:root:Epoch[49] Batch [740] Speed: 5.13 samples/sec Train-accuracy=0.965917
INFO:root:Epoch[49] Batch [750] Speed: 5.13 samples/sec Train-accuracy=0.958509
INFO:root:Epoch[49] Batch [760] Speed: 5.12 samples/sec Train-accuracy=0.965964
INFO:root:Epoch[49] Batch [770] Speed: 5.13 samples/sec Train-accuracy=0.962391
INFO:root:Saved checkpoint to "DeepLab-V2-0050.params"
INFO:root:                            --->Epoch[49] Train-accuracy=0.968494
INFO:root: in eval process...
INFO:root:batch[179] Validation-accuracy=0.837493
```

### 效果
原图
![原图](http://ww1.sinaimg.cn/large/6425ef91ly1fd5e8a7ulaj20eo0e8tle)
效果图
![网络输出](http://ww1.sinaimg.cn/large/6425ef91ly1fd5e6xfz9oj20e70e4aag)
