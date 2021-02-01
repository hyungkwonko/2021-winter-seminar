import time
import math
import torch 
import torch.nn as nn
from torch import optim
import torchvision
from a5_helper import *
import matplotlib.pyplot as plt


def hello_single_stage_detector():
    print("Hello from single_stage_detector.py!")


def GenerateAnchor(anc, grid):
  """
  Anchor generator.

  Inputs:
  - anc: Tensor of shape (A, 2) giving the shapes of anchor boxes to consider at
    each point in the grid. anc[a] = (w, h) gives the width and height of the
    a'th anchor shape.
  - grid: Tensor of shape (B, H', W', 2) giving the (x, y) coordinates of the
    center of each feature from the backbone feature map. This is the tensor
    returned from GenerateGrid.
  
  Outputs:
  - anchors: Tensor of shape (B, A, H', W', 4) giving the positions of all
    anchor boxes for the entire image. anchors[b, a, h, w] is an anchor box
    centered at grid[b, h, w], whose shape is given by anc[a]; we parameterize
    boxes as anchors[b, a, h, w] = (x_tl, y_tl, x_br, y_br), where (x_tl, y_tl)
    and (x_br, y_br) give the xy coordinates of the top-left and bottom-right
    corners of the box.
  """
  anchors = None
  ##############################################################################
  # TODO: Given a set of anchor shapes and a grid cell on the activation map,  #
  # generate all the anchor coordinates for each image. Support batch input.   #
  ##############################################################################
  # Replace "pass" statement with your code

  A, _ = anc.shape
  B, H_, W_, _ = grid.shape
  anchors = torch.zeros((B, A, H_, W_, 4), dtype=anc.dtype, device=anc.device)

  for i in range(A):
    w, h = anc[i]
    anchors[:, i, :, :, 0] = grid[:, :, :, 0] - w/2 # x top left
    anchors[:, i, :, :, 1] = grid[:, :, :, 1] - h/2 # y top left
    anchors[:, i, :, :, 2] = grid[:, :, :, 0] + w/2 # x bottom right
    anchors[:, i, :, :, 3] = grid[:, :, :, 1] + h/2 # y bottom right

  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################

  return anchors


def GenerateProposal(anchors, offsets, method='YOLO'):
  """
  Proposal generator.

  Inputs:
  - anchors: Anchor boxes, of shape (B, A, H', W', 4). Anchors are represented
    by the coordinates of their top-left and bottom-right corners.
  - offsets: Transformations of shape (B, A, H', W', 4) that will be used to
    convert anchor boxes into region proposals. The transformation
    offsets[b, a, h, w] = (tx, ty, tw, th) will be applied to the anchor
    anchors[b, a, h, w]. For YOLO, assume that tx and ty are in the range
    (-0.5, 0.5).
  - method: Which transformation formula to use, either 'YOLO' or 'FasterRCNN'
  
  Outputs:
  - proposals: Region proposals of shape (B, A, H', W', 4), represented by the
    coordinates of their top-left and bottom-right corners. Applying the
    transform offsets[b, a, h, w] to the anchor [b, a, h, w] should give the
    proposal proposals[b, a, h, w].
  
  """
  assert(method in ['YOLO', 'FasterRCNN'])
  proposals = None
  ##############################################################################
  # TODO: Given anchor coordinates and the proposed offset for each anchor,    #
  # compute the proposal coordinates using the transformation formulas above.  #
  ##############################################################################
  # Replace "pass" statement with your code

  B, A, H_, W_, _ = anchors.shape
  proposals = torch.zeros((B, A, H_, W_, 4), dtype=anchors.dtype, device=anchors.device)

  x_tl = anchors[:, :, :, :, 0]
  y_tl = anchors[:, :, :, :, 1]
  x_br = anchors[:, :, :, :, 2]
  y_br = anchors[:, :, :, :, 3]

  x_c = (x_tl + x_br) / 2
  y_c = (y_tl + y_br) / 2
  w_a = x_br - x_tl
  h_a = y_br - y_tl

  tx = offsets[:, :, :, :, 0]
  ty = offsets[:, :, :, :, 1]
  tw = offsets[:, :, :, :, 2]
  th = offsets[:, :, :, :, 3]

  if method == 'YOLO':
    tx = torch.clamp(tx, min=-0.5, max=0.5)
    ty = torch.clamp(ty, min=-0.5, max=0.5)
    proposals[:, :, :, :, 0] = x_c + tx
    proposals[:, :, :, :, 1] = y_c + ty
  else:
    proposals[:, :, :, :, 0] = x_c + tx * w_a
    proposals[:, :, :, :, 1] = y_c + ty * h_a
  proposals[:, :, :, :, 2] = w_a * tw.exp()
  proposals[:, :, :, :, 3] = h_a * th.exp()

  pro_cx = proposals[:, :, :, :, 0].clone()
  pro_cy = proposals[:, :, :, :, 1].clone()
  pro_w = proposals[:, :, :, :, 2].clone()
  pro_h = proposals[:, :, :, :, 3].clone()
  proposals[:, :, :, :, 0] = pro_cx - pro_w / 2
  proposals[:, :, :, :, 2] = pro_cx + pro_w / 2
  proposals[:, :, :, :, 1] = pro_cy - pro_h / 2
  proposals[:, :, :, :, 3] = pro_cy + pro_h / 2

  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################

  return proposals


def IoU(proposals, bboxes):
  """
  Compute intersection over union between sets of bounding boxes.

  Inputs:
  - proposals: Proposals of shape (B, A, H', W', 4)
  - bboxes: Ground-truth boxes from the DataLoader of shape (B, N, 5).
    Each ground-truth box is represented as tuple (x_lt, y_lt, x_rb, y_rb, class).
    If image i has fewer than N boxes, then bboxes[i] will be padded with extra
    rows of -1.
  
  Outputs:
  - iou_mat: IoU matrix of shape (B, A*H'*W', N) where iou_mat[b, i, n] gives
    the IoU between one element of proposals[b] and bboxes[b, n].

  For this implementation you DO NOT need to filter invalid proposals or boxes;
  in particular you don't need any special handling for bboxxes that are padded
  with -1.
  """
  iou_mat = None
  ##############################################################################
  # TODO: Compute the Intersection over Union (IoU) on proposals and GT boxes. #
  # No need to filter invalid proposals/bboxes (i.e., allow region area <= 0). #
  # However, you need to make sure to compute the IoU correctly (it should be  #
  # 0 in those cases.                                                          # 
  # You need to ensure your implementation is efficient (no for loops).        #
  # HINT:                                                                      #
  # IoU = Area of Intersection / Area of Union, where                          #
  # Area of Union = Area of Proposal + Area of BBox - Area of Intersection     #
  # and the Area of Intersection can be computed using the top-left corner and #
  # bottom-right corner of proposal and bbox. Think about their relationships. #
  ##############################################################################
  # Replace "pass" statement with your code

  B, A, H_, W_, _ = proposals.shape
  B, N, _ = bboxes.shape

  device = proposals.device
  dtype = proposals.dtype

  bboxes = bboxes.to(device)

  px_tl = proposals[:, :, :, :, 0].reshape(B, -1, 1)
  py_tl = proposals[:, :, :, :, 1].reshape(B, -1, 1)
  px_br = proposals[:, :, :, :, 2].reshape(B, -1, 1)
  py_br = proposals[:, :, :, :, 3].reshape(B, -1, 1)
  # px_c = (px_tl + px_br) / 2
  # py_c = (py_tl + py_br) / 2
  pw_a = px_br - px_tl
  ph_a = py_br - py_tl
  area_p = (pw_a * ph_a)  # (B, A*H_*W_, 1)

  bx_tl = bboxes[:, :, 0].reshape(B, 1, N)  # (B, 1, N)
  by_tl = bboxes[:, :, 1].reshape(B, 1, N)
  bx_br = bboxes[:, :, 2].reshape(B, 1, N)
  by_br = bboxes[:, :, 3].reshape(B, 1, N)
  # bx_c = (bx_tl + bx_br) / 2
  # by_c = (by_tl + by_br) / 2
  bw_a = bx_br - bx_tl
  bh_a = by_br - by_tl
  area_b = (bw_a * bh_a)  # (B, 1, N)

  area_u = area_p + area_b  # (B, A*H_*W_, N)

  x_val = torch.minimum(px_br, bx_br) - torch.maximum(px_tl, bx_tl)
  x_val_zero = torch.zeros(x_val.shape, device=device, dtype=dtype)
  y_val = torch.minimum(py_br, by_br) - torch.maximum(py_tl, by_tl)
  y_val_zero = torch.zeros(y_val.shape, device=device, dtype=dtype)

  area_i = torch.maximum(x_val, x_val_zero) * torch.maximum(y_val, y_val_zero)

  area_u -= area_i
  iou_mat = area_i / area_u

  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  return iou_mat


class PredictionNetwork(nn.Module):
  def __init__(self, in_dim, hidden_dim=128, num_anchors=9, num_classes=20, drop_ratio=0.3):
    super().__init__()

    assert(num_classes != 0 and num_anchors != 0)
    self.num_classes = num_classes
    self.num_anchors = num_anchors

    ##############################################################################
    # TODO: Set up a network that will predict outputs for all anchors. This     #
    # network should have a 1x1 convolution with hidden_dim filters, followed    #
    # by a Dropout layer with p=drop_ratio, a Leaky ReLU nonlinearity, and       #
    # finally another 1x1 convolution layer to predict all outputs. You can      #
    # use an nn.Sequential for this network, and store it in a member variable.  #
    # HINT: The output should be of shape (B, 5*A+C, 7, 7), where                #
    # A=self.num_anchors and C=self.num_classes.                                 #
    ##############################################################################
    # Make sure to name your prediction network pred_layer.

    # Replace "pass" statement with your code
    self.pred_layer = nn.Sequential(
      nn.Conv2d(in_dim, hidden_dim, (1, 1), stride=1),
      nn.Dropout2d(p=drop_ratio),
      nn.LeakyReLU(),
      nn.Conv2d(hidden_dim, self.num_anchors * 5 + self.num_classes, (1, 1), stride=1)
    )

    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

  def _extract_anchor_data(self, anchor_data, anchor_idx):
    """
    Inputs:
    - anchor_data: Tensor of shape (B, A, D, H, W) giving a vector of length
      D for each of A anchors at each point in an H x W grid.
    - anchor_idx: int64 Tensor of shape (M,) giving anchor indices to extract

    Returns:
    - extracted_anchors: Tensor of shape (M, D) giving anchor data for each
      of the anchors specified by anchor_idx.
    """
    B, A, D, H, W = anchor_data.shape
    anchor_data = anchor_data.permute(0, 1, 3, 4, 2).contiguous().view(-1, D)
    extracted_anchors = anchor_data[anchor_idx]
    return extracted_anchors
  
  def _extract_class_scores(self, all_scores, anchor_idx):
    """
    Inputs:
    - all_scores: Tensor of shape (B, C, H, W) giving classification scores for
      C classes at each point in an H x W grid.
    - anchor_idx: int64 Tensor of shape (M,) giving the indices of anchors at
      which to extract classification scores

    Returns:
    - extracted_scores: Tensor of shape (M, C) giving the classification scores
      for each of the anchors specified by anchor_idx.
    """
    B, C, H, W = all_scores.shape
    A = self.num_anchors
    all_scores = all_scores.contiguous().permute(0, 2, 3, 1).contiguous()
    all_scores = all_scores.view(B, 1, H, W, C).expand(B, A, H, W, C)
    all_scores = all_scores.reshape(B * A * H * W, C)
    extracted_scores = all_scores[anchor_idx]
    return extracted_scores

  def forward(self, features, pos_anchor_idx=None, neg_anchor_idx=None):
    """
    Run the forward pass of the network to predict outputs given features
    from the backbone network.

    Inputs:
    - features: Tensor of shape (B, in_dim, 7, 7) giving image features computed
      by the backbone network.
    - pos_anchor_idx: int64 Tensor of shape (M,) giving the indices of anchors
      marked as positive. These are only given during training; at test-time
      this should be None.
    - neg_anchor_idx: int64 Tensor of shape (M,) giving the indices of anchors
      marked as negative. These are only given at training; at test-time this
      should be None.
    
    The outputs from this method are different during training and inference.
    
    During training, pos_anchor_idx and neg_anchor_idx are given and identify
    which anchors should be positive and negative, and this forward pass needs
    to extract only the predictions for the positive and negative anchors.

    During inference, only features are provided and this method needs to return
    predictions for all anchors.

    Outputs (During training):
    - conf_scores: Tensor of shape (2*M, 1) giving the predicted classification
      scores for positive anchors and negative anchors (in that order).
    - offsets: Tensor of shape (M, 4) giving predicted transformation for
      positive anchors.
    - class_scores: Tensor of shape (M, C) giving classification scores for
      positive anchors.

    Outputs (During inference):
    - conf_scores: Tensor of shape (B, A, H, W) giving predicted classification
      scores for all anchors.
    - offsets: Tensor of shape (B, A, 4, H, W) giving predicted transformations
      all all anchors.
    - class_scores: Tensor of shape (B, C, H, W) giving classification scores for
      each spatial position.
    """
    conf_scores, offsets, class_scores = None, None, None
    ############################################################################
    # TODO: Use backbone features to predict conf_scores, offsets, and         #
    # class_scores. Make sure conf_scores is between 0 and 1 by squashing the  #
    # network output with a sigmoid. Also make sure the first two elements t^x #
    # and t^y of offsets are between -0.5 and 0.5 by squashing with a sigmoid  #
    # and subtracting 0.5.                                                     #
    #                                                                          #
    # During training you need to extract the outputs for only the positive    #
    # and negative anchors as specified above.                                 #
    #                                                                          #
    # HINT: You can use the provided helper methods self._extract_anchor_data  #
    # and self._extract_class_scores to extract information for positive and   #
    # negative anchors specified by pos_anchor_idx and neg_anchor_idx.         #
    ############################################################################
    # Replace "pass" statement with your code

    B, _, H, W = features.shape

    out = self.pred_layer(features)
    anchor_data = out[:, :5*self.num_anchors, :, :].reshape(B, self.num_anchors, -1, H, W) # (B, A, 5, H, W)
    class_scores = out[:, -self.num_classes:, :, :] # (B, C, H, W)
    offsets = anchor_data[:, :, 1:, :, :].clone() # (B, A, 4, H, W)
    offsets[:, :, :2, :, :] = torch.sigmoid(offsets[:, :, :2, :, :]) - 0.5
    conf_scores = torch.sigmoid(anchor_data[:, :, :1, :, :].clone()) # (B, A, 1, H, W)

    if pos_anchor_idx == None or neg_anchor_idx == None:
      conf_scores = conf_scores.reshape(B, self.num_anchors, H, W)
    else:
      offsets = self._extract_anchor_data(offsets, pos_anchor_idx)
      class_scores = self._extract_class_scores(class_scores, pos_anchor_idx)

      pos = self._extract_anchor_data(conf_scores, pos_anchor_idx) # (M, 1)
      neg = self._extract_anchor_data(conf_scores, neg_anchor_idx) # (M, 1)
      conf_scores = torch.cat([pos, neg]) # (2*M, 1)

    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return conf_scores, offsets, class_scores


class SingleStageDetector(nn.Module):
  def __init__(self):
    super().__init__()

    self.anchor_list = torch.tensor([[1., 1], [2, 2], [3, 3], [4, 4], [5, 5], [2, 3], [3, 2], [3, 5], [5, 3]]) # READ ONLY
    self.feat_extractor = FeatureExtractor()
    self.num_classes = 20
    self.pred_network = PredictionNetwork(1280, num_anchors=self.anchor_list.shape[0], \
                                          num_classes=self.num_classes)
  def forward(self, images, bboxes):
    """
    Training-time forward pass for the single-stage detector.

    Inputs:
    - images: Input images, of shape (B, 3, 224, 224)
    - bboxes: GT bounding boxes of shape (B, N, 5) (padded)

    Outputs:
    - total_loss: Torch scalar giving the total loss for the batch.
    """
    # weights to multiple to each loss term
    w_conf = 1 # for conf_scores
    w_reg = 1 # for offsets
    w_cls = 1 # for class_prob

    total_loss = None
    ##############################################################################
    # TODO: Implement the forward pass of SingleStageDetector.                   #
    # A few key steps are outlined as follows:                                   #
    # i) Image feature extraction,                                               #
    # ii) Grid and anchor generation,                                            #
    # iii) Compute IoU between anchors and GT boxes and then determine activated/#
    #      negative anchors, and GT_conf_scores, GT_offsets, GT_class,           #
    # iv) Compute conf_scores, offsets, class_prob through the prediction network#
    # v) Compute the total_loss which is formulated as:                          #
    #    total_loss = w_conf * conf_loss + w_reg * reg_loss + w_cls * cls_loss,  #
    #    where conf_loss is determined by ConfScoreRegression, w_reg by          #
    #    BboxRegression, and w_cls by ObjectClassification.                      #
    # HINT: Set `neg_thresh=0.2` in ReferenceOnActivatedAnchors in this notebook #
    #       (A5-1) for a better performance than with the default value.         #
    ##############################################################################
    # Replace "pass" statement with your code

    device = images.device

    # feature extraction
    features = self.feat_extractor(images)

    # grid and anchor generation
    grid_list = GenerateGrid(batch_size=images.shape[0])
    anc_list = GenerateAnchor(self.anchor_list, grid_list)

    anc_list = anc_list.to(device)

    # compute IoU and determine requirements
    iou_mat = IoU(anc_list, bboxes)
    pos_anchor_idx, neg_anchor_idx, GT_conf_scores, GT_offsets, GT_class, _, _ = ReferenceOnActivatedAnchors(anc_list, \
        bboxes, grid_list, iou_mat, neg_thresh=0.2, method='YOLO')

    conf_scores, offsets, class_prob = self.pred_network(features, pos_anchor_idx, neg_anchor_idx)

    _, A, H, W, _ = anc_list.shape

    conf_loss = ConfScoreRegression(conf_scores, GT_conf_scores)
    reg_loss = BboxRegression(offsets, GT_offsets)
    cls_loss = ObjectClassification(class_prob, GT_class, images.shape[0], A * H * W, pos_anchor_idx)

    total_loss = w_conf * conf_loss + w_reg * reg_loss + w_cls * cls_loss

    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return total_loss
  
  def inference(self, images, thresh=0.5, nms_thresh=0.7):
    """"
    Inference-time forward pass for the single stage detector.

    Inputs:
    - images: Input images
    - thresh: Threshold value on confidence scores
    - nms_thresh: Threshold value on NMS

    Outputs:
    - final_propsals: Keeped proposals after confidence score thresholding and NMS,
                      a list of B (*x4) tensors
    - final_conf_scores: Corresponding confidence scores, a list of B (*x1) tensors
    - final_class: Corresponding class predictions, a list of B  (*x1) tensors
    """
    final_proposals, final_conf_scores, final_class = [], [], []
    ##############################################################################
    # TODO: Predicting the final proposal coordinates `final_proposals`,         #
    # confidence scores `final_conf_scores`, and the class index `final_class`.  #
    # The overall steps are similar to the forward pass but now you do not need  #
    # to decide the activated nor negative anchors.                              #
    # HINT: Thresholding the conf_scores based on the threshold value `thresh`.  #
    # Then, apply NMS (torchvision.ops.nms) to the filtered proposals given the  #
    # threshold `nms_thresh`.                                                    #
    # The class index is determined by the class with the maximal probability.   #
    # Note that `final_propsals`, `final_conf_scores`, and `final_class` are all #
    # lists of B 2-D tensors (you may need to unsqueeze dim=1 for the last two). #
    ##############################################################################
    # Replace "pass" statement with your code

    with torch.no_grad():

      device = images.device
      # dtype = images.dtype  # not used

      # feature extraction
      features = self.feat_extractor(images)

      # grid and anchor generation
      grid_list = GenerateGrid(batch_size=images.shape[0])
      anc_list = GenerateAnchor(self.anchor_list, grid_list).to(device)

      conf_scores, offsets, class_prob = self.pred_network(features)
      offsets = offsets.permute(0, 1, 3, 4, 2)

      proposals = GenerateProposal(anc_list, offsets, method='YOLO')

      # print(grid_list.shape)
      # print(anc_list.shape)     # (B, A, H, W, 4)
      # print(proposals.shape)    # (B, A, H, W, 4)
      # print(features.shape)     # (B, 1280, H, W)
      # print(conf_scores.shape)  # (B, A, H, W)
      # print(offsets.shape)      # (B, A, H, W, 4)
      # print(class_prob.shape)   # (B, C, H, W)

      (B, A, H, W) = conf_scores.shape

      conf_scores = conf_scores.reshape(B, -1)     # (B, A*H*W)
      offsets = offsets.reshape(B, -1, 4)          # (B, A*H*W, 4)
      proposals = proposals.reshape(B, -1, 4)      # (B, A*H*W, 4)
      class_prob = class_prob.permute(0, 2, 3, 1)  # (B, H, W, C)

      for i in range(B):
        index = torch.zeros(conf_scores[i].shape, device=device, dtype=torch.bool)
        index[conf_scores[i] > thresh] = True
        tmp_conf_scores = conf_scores[i, index].clone()
        tmp_proposals = proposals[i, index, :].clone()
        tmp_class_prob = class_prob[i].argmax(dim=2).expand(A, H, W).reshape(-1)[index].clone()

        keep = torchvision.ops.nms(tmp_proposals, tmp_conf_scores, iou_threshold=nms_thresh)
        final_proposals.append(tmp_proposals[keep])
        final_conf_scores.append(tmp_conf_scores[keep].unsqueeze(1))
        final_class.append(tmp_class_prob[keep].unsqueeze(1))

    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return final_proposals, final_conf_scores, final_class


def nms(boxes, scores, iou_threshold=0.5, topk=None):
  """
  Non-maximum suppression removes overlapping bounding boxes.

  Inputs:
  - boxes: top-left and bottom-right coordinate values of the bounding boxes
    to perform NMS on, of shape Nx4
  - scores: scores for each one of the boxes, of shape N
  - iou_threshold: discards all overlapping boxes with IoU > iou_threshold; float
  - topk: If this is not None, then return only the topk highest-scoring boxes.
    Otherwise if this is None, then return all boxes that pass NMS.

  Outputs:
  - keep: torch.long tensor with the indices of the elements that have been
    kept by NMS, sorted in decreasing order of scores; of shape [num_kept_boxes]
  """

  if (not boxes.numel()) or (not scores.numel()):
    return torch.zeros(0, dtype=torch.long)

  keep = None
  #############################################################################
  # TODO: Implement non-maximum suppression which iterates the following:     #
  #       1. Select the highest-scoring box among the remaining ones,         #
  #          which has not been chosen in this step before                    #
  #       2. Eliminate boxes with IoU > threshold                             #
  #       3. If any boxes remain, GOTO 1                                      #
  #       Your implementation should not depend on a specific device type;    #
  #       you can use the device of the input if necessary.                   #
  # HINT: You can refer to the torchvision library code:                      #
  #   github.com/pytorch/vision/blob/master/torchvision/csrc/cpu/nms_cpu.cpp  #
  #############################################################################
  # Replace "pass" statement with your code

  boxes = boxes.to('cpu')
  scores = scores.to('cpu')

  val, idx = scores.sort(descending=True)
  idx_change = idx.clone()

  def small_IoU(box_i, box_j):

    device = box_i.device
    dtype = box_i.dtype

    pw_a = box_i[2] - box_i[0]
    ph_a = box_i[3] - box_i[1]
    area_p = (pw_a * ph_a)

    bw_a = box_j[2] - box_j[0]
    bh_a = box_j[3] - box_j[1]
    area_b = (bw_a * bh_a)

    area_u = area_p + area_b

    x_val = torch.minimum(box_i[2], box_j[2]) - torch.maximum(box_i[0], box_j[0])
    x_val_zero = torch.zeros(x_val.shape, device=device, dtype=dtype)
    y_val = torch.minimum(box_i[3], box_j[3]) - torch.maximum(box_i[1], box_j[1])
    y_val_zero = torch.zeros(y_val.shape, device=device, dtype=dtype)

    area_i = torch.maximum(x_val, x_val_zero) * torch.maximum(y_val, y_val_zero)
    area_u -= area_i

    return area_i / area_u

  for i, ix in enumerate(idx):
    if idx_change[i] == -1:
      continue
    for j, jx in enumerate(idx):
      if (j == i) or (idx_change[j] == -1):
        continue
      score = small_IoU(boxes[ix], boxes[jx])
      if score > iou_threshold:
        idx_change[j] = -1

  keep = idx[idx_change >= 0]

  if topk != None:
    keep = keep[:topk]

  #############################################################################
  #                              END OF YOUR CODE                             #
  #############################################################################
  return keep

def ConfScoreRegression(conf_scores, GT_conf_scores):
  """
  Use sum-squared error as in YOLO

  Inputs:
  - conf_scores: Predicted confidence scores
  - GT_conf_scores: GT confidence scores
  
  Outputs:
  - conf_score_loss
  """
  # the target conf_scores for negative samples are zeros
  GT_conf_scores = torch.cat((torch.ones_like(GT_conf_scores), \
                              torch.zeros_like(GT_conf_scores)), dim=0).view(-1, 1)
  conf_score_loss = torch.sum((conf_scores - GT_conf_scores)**2) * 1. / GT_conf_scores.shape[0]
  return conf_score_loss


def BboxRegression(offsets, GT_offsets):
  """"
  Use sum-squared error as in YOLO
  For both xy and wh

  Inputs:
  - offsets: Predicted box offsets
  - GT_offsets: GT box offsets
  
  Outputs:
  - bbox_reg_loss
  """
  bbox_reg_loss = torch.sum((offsets - GT_offsets)**2) * 1. / GT_offsets.shape[0]
  return bbox_reg_loss

