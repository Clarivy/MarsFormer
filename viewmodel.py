import torch
#pthfile = "checkpoints/save/performer_test_fast_attention_neg0_baseline2/200_model.pth"
#pthfile = "checkpoints/save/performer_test_fast_attention_neg0_causal2/65_model.pth"          #.pth文件的路径
pthfile = "checkpoints/save/performer_test_fast_attention_neg0_causal_4_perself_nnmul/165_model.pth"
model = torch.load(pthfile, torch.device('cuda'))    #设置在cpu环境下查询
print('type:')
print(type(model))  #查看模型字典长度
print('length:')
print(len(model))
print('key:')
for k in model.keys():  #查看模型字典里面的key
    print(k)
print('value:')
for k in model:         #查看模型字典里面的value
    print(k,model[k].size())
