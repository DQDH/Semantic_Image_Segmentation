import numpy as np
def make_cue_pickle(out,label_src):
    label_index = np.where(label_src == 1)[0] + 1
    label_index=np.append(0,label_index)
    cues_i = ()
    for label_id in label_index:
        if label_id == 0:
            rc_index = np.where(out == label_id)
            rc_num = np.zeros([1, rc_index[0].size])
            rc_num = tuple(rc_num)
            cues_i = rc_num + rc_index
        else:
            rc_index = np.where(out == label_id)
            rc_num = np.ones([rc_index[0].size]) * label_id
            cues_i = list(cues_i)
            cues_i[0] = tuple(np.array(cues_i[0], dtype=np.int64)) + tuple(np.array(rc_num, dtype=np.int64))
            cues_i[1] = tuple(cues_i[1]) + tuple(rc_index[0])
            cues_i[2] = tuple(cues_i[2]) + tuple(rc_index[1])
    cues_i[0] = np.array(cues_i[0])
    cues_i[1] = np.array(cues_i[1])
    cues_i[2] = np.array(cues_i[2])
    cues_i=tuple(cues_i)
    return cues_i
