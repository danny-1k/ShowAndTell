import torch

def dataloader_collate_fn(data):
    """
    Creates padded caption batch from batch of irregular sequences

    Args:
        data : (x, y) tuple from dataset, x is the image and y is the image caption.

    Returns:
        images: images in the batch
        caption_padded: the padded batch
        lenghts: lengths of the sequences in the batch
    """
    data.sort(key=lambda x: len(x[1]), reverse=True) # sort by caption length
    images, captions = zip(*data)
    images = torch.stack(images, 0)
    lengths = [len(caption) for caption in captions]

    max_length = max(lengths)
    batch_size = len(lengths)

    caption_padded = torch.zeros(batch_size,max_length)

    for i, cap in enumerate(captions):
        end = lengths[i]
        caption_padded[i, :end] = cap[:end]

    return images, caption_padded, lengths