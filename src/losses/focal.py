from catalyst.contrib.criterion import FocalLossBinary


class FocalLossMultiChannel(FocalLossBinary):
    """
    Compute focal loss for multi-class problem.
    Ignores targets having -1 label
    """

    def forward(self, logits, targets):
        """
        Args:
            logits:  [bs; num_classes; ...]
            targets: [bs; num_classes; ...]
        """
        num_classes = logits.size(1)
        loss = 0
        targets = targets.view(-1, num_classes)
        logits = logits.view(-1, num_classes)

        # Filter anchors with -1 label from loss computation
        if self.ignore is not None:
            not_ignored = targets != self.ignore

        for cls_idx in range(num_classes):
            cls_label_target = targets[..., cls_idx]
            cls_label_input = logits[..., cls_idx]
            if self.ignore is not None:
                cls_label_target = cls_label_target[not_ignored]
                cls_label_input = cls_label_input[not_ignored]
            loss += self.loss_fn(cls_label_input, cls_label_target)
        
        return loss
