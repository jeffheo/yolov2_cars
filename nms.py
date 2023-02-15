import torch 

def nms(out, batch_size): # 25 * 7 * 7 tensor
	pos, size, conf = out
	xywh = torch.cat([pos, size], 2)
	lrud = xywhTOlrud(xywh)
	lrud = torch.cat((lrud, conf), 2)
	conf_sort = torch.sort(lrud[:,:,4], dim=1, descending=True)

	finalist = []
	for img in range(batch_size):
		finalist.append([])
		lrud_img = lrud[img,:,:]
		conf_img = conf_sort[img,:]

		while conf_img.size() > 0:
			
			best_idx = conf_sort.indices[img,0]
			best_img = lrud_img[best_idx,0:3]
			finalist[img].append(best_img)
			rest_img = lrud_img[conf_sort.indices[img,1:],0:3]

			ious = iou(best_img, rest_img)
			indices = torch.nonzero(ious < 0.8)
			conf_img = conf_img[indices]

	return finalist