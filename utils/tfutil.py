import tensorflow as tf

def set_gpu_options(visible_idxs=[0], memory_growth=True):
    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    vgpus = [gpus[i] for i in visible_idxs if i<len(gpus)]
    tf.config.experimental.set_visible_devices(devices=vgpus, device_type='GPU')
    for gpu in vgpus:
        tf.config.experimental.set_memory_growth(gpu, memory_growth)

