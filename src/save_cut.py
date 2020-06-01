def save_cut_model(model, params):
    print(model)
    model.load_state_dict(t.load(path), strict=False)
    for module in model.named_modules():
        print(module)

    save_dic = {}
    for name, param in model.named_parameters():
        # print()
        # t.FloatTensor(param)
        save_nozero = []
        zero = t.zeros_like(param)
        print(type(param))
        param = t.where(param < 0.02, zero, param)
        No_zero = t.nonzero(param)
        No_zero = No_zero.numpy()
        print(No_zero.tolist())
        print(name, type(param))
        for local in No_zero:
            temp = param
            for i in local:
                temp = temp[i]
            save_nozero.append(temp)
        save_param = (No_zero, temp)
        
    save_dic[name] = save_param

    torch.save(save_dic,params.cut_path)