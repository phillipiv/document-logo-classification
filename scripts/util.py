
def preprocess_document(document):

    w, h, _ = document.shape

    return document[:int(h / 3), :, :]


