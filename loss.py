import numpy as np

def xywhTOlrud(box):
      x, y, w, h = box[0], box[1], box[2], box[3]
      l = x - w / 2
      r = x + w / 2
      u = y + h / 2
      d = 7 - h / 2
      return (l, r, u, d)

def iou(box1, box2):
	# box1 and box2: (l, r, u, d)
	
	li = max(box1[0], box2[0])
	ri = min(box1[1], box2[1])
	ui = min(box1[2], box2[2])
	di = max(box1[3], box2[3])
	
	ai = 0 # area of intersection
	if li < ri and di < ui:
		ai = (ri - li) * (ui - di)
	a1 = (box1[1] - box1[0]) * (box1[2] - box1[3])
	a2 = (box2[1] - box2[0]) * (box2[2] - box2[3])
	au = a1 + a2 - ai # area of union

	return ai / au

def loss(pred, gt, ld_coor, ld_noob, img_size):
	#pred is of shape [50, 7, 7], each of the 7 x 7 grid has 5 anchor boxes (confidence + 4 dim + 5 class)
    #gt is (x_min, x_max, y_min, y_max, class)
    gt_x_min, gt_x_max, gt_y_min, gt_y_max, gt_class = gt
    loss = 0
    gt_xc = (gt[0] + gt[1]) / 2
    gt_yc = (gt[2] + gt[3]) / 2
    gt_w = gt_x_max - gt_x_min
    gt_h = gt_y_max - gt_y_min
    dims = 7
    x_increment = img_size[0] / 7
    y_increment = img_size[1] / 7
    for i in range(dims): 
         for j in range(dims): 
            curr_vec = pred[:, i, j]
	        # center is inside bounding box (i, j)
            
            if gt_xc >= i * x_increment and gt_xc < (i + 1) * x_increment and  gt_yc >= i * y_increment and gt_yc < (i + 1) * y_increment:
			    #do something
                correct_anchor = 0
                max_IOU = 0
                for anchor in range(5):
                    curr_IOU = iou(gt[0:4], [xywhTOlrud(curr_vec[anchor * 10 + 1 : anchor * 10 + 5])])
                    if curr_IOU > max_IOU:
                        max_IOU = curr_IOU
                        correct_anchor = anchor
                anchor_pred = curr_vec[anchor * 10 : anchor * 10 + 10]
                center_loss = np.square(gt_xc - anchor_pred[1]) + np.square(gt_yc - anchor_pred[2])
                center_loss *= ld_coor
                wh_loss = np.square(np.sqrt(gt_w) - np.sqrt(anchor_pred[3])) + np.square(np.sqrt(gt_h) - np.sqrt(anchor_pred[4]))
                wh_loss *= ld_coor
                conf_loss = np.square(max_IOU - anchor_pred[0])
                class_loss = - np.log(anchor_pred[5 + gt_class - 1])
                loss += center_loss + wh_loss + conf_loss + class_loss
                loss -= ld_noob * np.square(curr_vec[10 * correct_anchor])
            
            loss += ld_noob * np.sum((np.square(np.take(curr_vec, [0, 10, 20, 30, 40]))))

    return loss
                