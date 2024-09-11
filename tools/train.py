import os,torch,json





def train(model, data, epoch, optimizer, scaler, scheduler, args, global_trained_steps, teacher_model=None):