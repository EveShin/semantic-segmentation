def Poly_Scheduler(optimizer, base_lr, curr_iter, max_iter, power=0.9):
    lr = base_lr * (1 - float(curr_iter) / max_iter) ** power

    return lr