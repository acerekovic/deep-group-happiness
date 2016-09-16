import cv2
import numpy as np
import tensorflow as tf

from utils.tensorbox import googlenet_load
from utils.tensorbox.annolist import AnnotationLib as al
from utils.tensorbox.rect import Rect
import json

def build_lstm_inner(lstm_input, H):
    '''
    build lstm decoder
    '''
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(H['arch']['lstm_size'], forget_bias=0.0)
    if H['arch']['num_lstm_layers'] > 1:
        lstm = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * H['arch']['num_lstm_layers'])
    else:
        lstm = lstm_cell

    batch_size = H['arch']['batch_size'] * H['arch']['grid_height'] * H['arch']['grid_width']
    state = tf.zeros([batch_size, lstm.state_size])

    outputs = []
    with tf.variable_scope('RNN', initializer=tf.random_uniform_initializer(-0.1, 0.1)):
        for time_step in range(H['arch']['rnn_len']):
            if time_step > 0: tf.get_variable_scope().reuse_variables()
            output, state = lstm(lstm_input, state)
            outputs.append(output)
    return outputs


def build_overfeat_inner(lstm_input, H):
    '''
    build simple overfeat decoder
    '''
    if H['arch']['rnn_len'] > 1:
        raise ValueError('rnn_len > 1 only supported with use_lstm == True')
    outputs = []
    with tf.variable_scope('Overfeat', initializer=tf.random_uniform_initializer(-0.1, 0.1)):
        w = tf.get_variable('ip', shape=[1024, H['arch']['lstm_size']])
        outputs.append(tf.matmul(lstm_input, w))
    return outputs


def to_idx(vec, w_shape):
    '''
    vec = (idn, idh, idw)
    w_shape = [n, h, w, c]
    '''
    return vec[:, 2] + w_shape[2] * (vec[:, 1] + w_shape[1] * vec[:, 0])


def interp(w, i, channel_dim):
    '''
    Input:
        w: A 4D block tensor of shape (n, h, w, c)
        i: A list of 3-tuples [(x_1, y_1, z_1), (x_2, y_2, z_2), ...],
            each having type (int, float, float)

        The 4D block represents a batch of 3D image feature volumes with c channels.
        The input i is a list of points  to index into w via interpolation. Direct
        indexing is not possible due to y_1 and z_1 being float values.
    Output:
        A list of the values: [
            w[x_1, y_1, z_1, :]
            w[x_2, y_2, z_2, :]
            ...
            w[x_k, y_k, z_k, :]
        ]
        of the same length == len(i)
    '''
    w_as_vector = tf.reshape(w, [-1, channel_dim])  # gather expects w to be 1-d
    upper_l = tf.to_int32(tf.concat(1, [i[:, 0:1], tf.floor(i[:, 1:2]), tf.floor(i[:, 2:3])]))
    upper_r = tf.to_int32(tf.concat(1, [i[:, 0:1], tf.floor(i[:, 1:2]), tf.ceil(i[:, 2:3])]))
    lower_l = tf.to_int32(tf.concat(1, [i[:, 0:1], tf.ceil(i[:, 1:2]), tf.floor(i[:, 2:3])]))
    lower_r = tf.to_int32(tf.concat(1, [i[:, 0:1], tf.ceil(i[:, 1:2]), tf.ceil(i[:, 2:3])]))

    upper_l_idx = to_idx(upper_l, tf.shape(w))
    upper_r_idx = to_idx(upper_r, tf.shape(w))
    lower_l_idx = to_idx(lower_l, tf.shape(w))
    lower_r_idx = to_idx(lower_r, tf.shape(w))

    upper_l_value = tf.gather(w_as_vector, upper_l_idx)
    upper_r_value = tf.gather(w_as_vector, upper_r_idx)
    lower_l_value = tf.gather(w_as_vector, lower_l_idx)
    lower_r_value = tf.gather(w_as_vector, lower_r_idx)

    alpha_lr = tf.expand_dims(i[:, 2] - tf.floor(i[:, 2]), 1)
    alpha_ud = tf.expand_dims(i[:, 1] - tf.floor(i[:, 1]), 1)

    upper_value = (1 - alpha_lr) * upper_l_value + (alpha_lr) * upper_r_value
    lower_value = (1 - alpha_lr) * lower_l_value + (alpha_lr) * lower_r_value
    value = (1 - alpha_ud) * upper_value + (alpha_ud) * lower_value
    return value


def bilinear_select(H, pred_boxes, early_feat, early_feat_channels, w_offset, h_offset):
    '''
    Function used for rezooming high level feature maps. Uses bilinear interpolation
    to select all channels at index (x, y) for a high level feature map, where x and y are floats.
    '''
    grid_size = H['arch']['grid_width'] * H['arch']['grid_height']
    outer_size = grid_size * H['arch']['batch_size']

    fine_stride = 8.  # pixels per 60x80 grid cell in 480x640 image
    coarse_stride = H['arch']['region_size']  # pixels per 15x20 grid cell in 480x640 image
    batch_ids = []
    x_offsets = []
    y_offsets = []
    for n in range(H['arch']['batch_size']):
        for i in range(H['arch']['grid_height']):
            for j in range(H['arch']['grid_width']):
                for k in range(H['arch']['rnn_len']):
                    batch_ids.append([n])
                    x_offsets.append([coarse_stride / 2. + coarse_stride * j])
                    y_offsets.append([coarse_stride / 2. + coarse_stride * i])

    batch_ids = tf.constant(batch_ids)
    x_offsets = tf.constant(x_offsets)
    y_offsets = tf.constant(y_offsets)

    pred_boxes_r = tf.reshape(pred_boxes, [outer_size * H['arch']['rnn_len'], 4])
    scale_factor = coarse_stride / fine_stride  # scale difference between 15x20 and 60x80 features

    pred_x_center = (pred_boxes_r[:, 0:1] + w_offset * pred_boxes_r[:, 2:3] + x_offsets) / fine_stride
    pred_x_center_clip = tf.clip_by_value(pred_x_center,
                                          0,
                                          scale_factor * H['arch']['grid_width'] - 1)
    pred_y_center = (pred_boxes_r[:, 1:2] + h_offset * pred_boxes_r[:, 3:4] + y_offsets) / fine_stride
    pred_y_center_clip = tf.clip_by_value(pred_y_center,
                                          0,
                                          scale_factor * H['arch']['grid_height'] - 1)

    interp_indices = tf.concat(1, [tf.to_float(batch_ids), pred_y_center_clip, pred_x_center_clip])
    return interp_indices


def rezoom(H, pred_boxes, early_feat, early_feat_channels, w_offsets, h_offsets):
    '''
    Rezoom into a feature map at multiple interpolation points in a grid.

    If the predicted object center is at X, len(w_offsets) == 3, and len(h_offsets) == 5,
    the rezoom grid will look as follows:

    [o o o]
    [o o o]
    [o X o]
    [o o o]
    [o o o]

    Where each letter indexes into the feature map with bilinear interpolation
    '''

    grid_size = H['arch']['grid_width'] * H['arch']['grid_height']
    outer_size = grid_size * H['arch']['batch_size']
    indices = []
    for w_offset in w_offsets:
        for h_offset in h_offsets:
            indices.append(bilinear_select(H, pred_boxes, early_feat, early_feat_channels, w_offset, h_offset))

    interp_indices = tf.concat(0, indices)
    rezoom_features = interp(early_feat, interp_indices, early_feat_channels)
    rezoom_features_r = tf.reshape(rezoom_features,
                                   [len(w_offsets) * len(h_offsets), outer_size, H['arch']['rnn_len'],
                                    early_feat_channels])
    rezoom_features_t = tf.transpose(rezoom_features_r, [1, 2, 0, 3])
    rezoom_features_t_r = tf.reshape(rezoom_features_t,
                                     [outer_size, H['arch']['rnn_len'],
                                      len(w_offsets) * len(h_offsets) * early_feat_channels])

    return rezoom_features_t_r


def build_forward(H, x, googlenet, phase, reuse):
    '''
    Construct the forward model
    '''

    grid_size = H['arch']['grid_width'] * H['arch']['grid_height']
    outer_size = grid_size * H['arch']['batch_size']
    input_mean = 117.
    x -= input_mean
    global early_feat
    Z, early_feat, _ = googlenet_load.model(x, googlenet, H)
    early_feat_channels = H['arch']['early_feat_channels']
    early_feat = early_feat[:, :, :, :early_feat_channels]

    if H['arch']['avg_pool_size'] > 1:
        pool_size = H['arch']['avg_pool_size']
        Z1 = Z[:, :, :, :700]
        Z2 = Z[:, :, :, 700:]
        Z2 = tf.nn.avg_pool(Z2, ksize=[1, pool_size, pool_size, 1], strides=[1, 1, 1, 1], padding='SAME')
        Z = tf.concat(3, [Z1, Z2])
    Z = tf.reshape(Z, [H['arch']['batch_size'] * H['arch']['grid_width'] * H['arch']['grid_height'], 1024])

    with tf.variable_scope('decoder', reuse=reuse):
        scale_down = 0.01
        lstm_input = tf.reshape(Z * scale_down, (H['arch']['batch_size'] * grid_size, 1024))
        if H['arch']['use_lstm']:
            lstm_outputs = build_lstm_inner(lstm_input, H)
        else:
            lstm_outputs = build_overfeat_inner(lstm_input, H)

        pred_boxes = []
        pred_logits = []
        initializer = tf.random_uniform_initializer(-0.1, 0.1)
        for k in range(H['arch']['rnn_len']):
            output = lstm_outputs[k]
            if phase == 'train':
                output = tf.nn.dropout(output, 0.5)
            box_weights = tf.get_variable('box_ip%d' % k,
                                          shape=(H['arch']['lstm_size'], 4),
                                          initializer=initializer)
            conf_weights = tf.get_variable('conf_ip%d' % k,
                                           shape=(H['arch']['lstm_size'], 2),
                                           initializer=initializer)

            pred_boxes_step = tf.reshape(tf.matmul(output, box_weights) * 50,
                                         [outer_size, 1, 4])

            pred_boxes.append(pred_boxes_step)
            pred_logits.append(tf.reshape(tf.matmul(output, conf_weights),
                                          [outer_size, 1, 2]))

        pred_boxes = tf.concat(1, pred_boxes)
        pred_logits = tf.concat(1, pred_logits)
        pred_logits_squash = tf.reshape(pred_logits,
                                        [outer_size * H['arch']['rnn_len'], 2])
        pred_confidences_squash = tf.nn.softmax(pred_logits_squash)
        pred_confidences = tf.reshape(pred_confidences_squash,
                                      [outer_size, H['arch']['rnn_len'], 2])

        if H['arch']['use_rezoom']:
            pred_confs_deltas = []
            pred_boxes_deltas = []
            w_offsets = H['arch']['rezoom_w_coords']
            h_offsets = H['arch']['rezoom_h_coords']
            num_offsets = len(w_offsets) * len(h_offsets)
            rezoom_features = rezoom(H, pred_boxes, early_feat, early_feat_channels, w_offsets, h_offsets)
            if phase == 'train':
                rezoom_features = tf.nn.dropout(rezoom_features, 0.5)
            for k in range(H['arch']['rnn_len']):
                delta_features = tf.concat(1, [lstm_outputs[k], rezoom_features[:, k, :] / 1000.])
                dim = 128
                delta_weights1 = tf.get_variable(
                    'delta_ip1%d' % k,
                    shape=[H['arch']['lstm_size'] + early_feat_channels * num_offsets, dim],
                    initializer=initializer)
                # TODO: add dropout here ?
                ip1 = tf.nn.relu(tf.matmul(delta_features, delta_weights1))
                if phase == 'train':
                    ip1 = tf.nn.dropout(ip1, 0.5)
                delta_confs_weights = tf.get_variable(
                    'delta_ip2%d' % k,
                    shape=[dim, 2],
                    initializer=initializer)
                if H['arch']['reregress']:
                    delta_boxes_weights = tf.get_variable(
                        'delta_ip_boxes%d' % k,
                        shape=[dim, 4],
                        initializer=initializer)
                    pred_boxes_deltas.append(tf.reshape(tf.matmul(ip1, delta_boxes_weights) * 5,
                                                        [outer_size, 1, 4]))
                scale = H['arch'].get('rezoom_conf_scale', 50)
                pred_confs_deltas.append(tf.reshape(tf.matmul(ip1, delta_confs_weights) * scale,
                                                    [outer_size, 1, 2]))
            pred_confs_deltas = tf.concat(1, pred_confs_deltas)
            if H['arch']['reregress']:
                pred_boxes_deltas = tf.concat(1, pred_boxes_deltas)
            summary_op = tf.merge_all_summaries()
            return pred_boxes, pred_logits, pred_confidences, pred_confs_deltas, pred_boxes_deltas, rezoom_features

    return pred_boxes, pred_logits, pred_confidences


def add_rectangles(H, orig_image, confidences, boxes, arch, use_stitching=False, rnn_len=1, min_conf=0.5, tau=0.25):

    image = np.copy(orig_image[0])
    boxes_r = np.reshape(boxes, (-1,
                                 arch["grid_height"],
                                 arch["grid_width"],
                                 rnn_len,
                                 4))
    confidences_r = np.reshape(confidences, (-1,
                                             arch["grid_height"],
                                             arch["grid_width"],
                                             rnn_len,
                                             2))
    cell_pix_size = H['arch']['region_size']
    all_rects = [[[] for _ in range(arch["grid_width"])] for _ in range(arch["grid_height"])]
    for n in range(0, H['arch']['rnn_len']):
        for y in range(arch["grid_height"]):
            for x in range(arch["grid_width"]):
                bbox = boxes_r[0, y, x, n, :]
                conf = confidences_r[0, y, x, n, 1]
                abs_cx = int(bbox[0]) + cell_pix_size/2 + cell_pix_size * x
                abs_cy = int(bbox[1]) + cell_pix_size/2 + cell_pix_size * y
                h = max(1, bbox[3])
                w = max(1, bbox[2])
                #w = h * 0.4
                all_rects[y][x].append(Rect(abs_cx,abs_cy,w,h,conf))

    if use_stitching:
        from utils.tensorbox.stitch_wrapper import stitch_rects
        acc_rects = stitch_rects(all_rects, tau)
    else:
        acc_rects = [r for row in all_rects for cell in row for r in cell if r.confidence > 0.1]


    for rect in acc_rects:
        if rect.confidence > min_conf:
            cv2.rectangle(image,
                (rect.cx-int(rect.width/2), rect.cy-int(rect.height/2)),
                (rect.cx+int(rect.width/2), rect.cy+int(rect.height/2)),
                (0,255,0),
                2)

    rects = []
    for rect in acc_rects:
        r = al.AnnoRect()
        r.x1 = rect.cx - rect.width/2.
        r.x2 = rect.cx + rect.width/2.
        r.y1 = rect.cy - rect.height/2.
        r.y2 = rect.cy + rect.height/2.
        r.score = rect.true_confidence
        rects.append(r)

    return image, rects

def init_model(hypes_file,iteration):
    # Now, read a model
    with tf.name_scope('initialize_face_detector'):
        #hypes_file = './data/models/face-detector-hypes.json'
        #iteration = 25000

        with open(hypes_file, 'r') as f:
            H = json.load(f)

        # Init stored graph
        tf.reset_default_graph()
        googlenet = googlenet_load.init(H)
        x_in = tf.placeholder(tf.float32, name='x_in', shape=[H['arch']['image_height'], H['arch']['image_width'], 3])
        if H['arch']['use_rezoom']:
            pred_boxes, pred_logits, pred_confidences, pred_confs_deltas, pred_boxes_deltas, rezoom_features = build_forward(
                H,
                tf.expand_dims(
                    x_in,
                    0),
                googlenet,
                'test',
                reuse=None)
            grid_area = H['arch']['grid_height'] * H['arch']['grid_width']
            pred_confidences = tf.reshape(
                tf.nn.softmax(tf.reshape(pred_confs_deltas, [grid_area * H['arch']['rnn_len'], 2])),
                [grid_area, H['arch']['rnn_len'], 2])
            if H['arch']['reregress']:
                pred_boxes = pred_boxes + pred_boxes_deltas

            return x_in, pred_boxes, pred_logits, pred_confidences, pred_confs_deltas, pred_boxes_deltas, rezoom_features
        else:
            pred_boxes, pred_logits, pred_confidences = build_forward(H, tf.expand_dims(x_in, 0), googlenet, 'test',
                                                                      reuse=None)


            return x_in, pred_boxes, pred_logits, pred_confidences, 0, 0, 0