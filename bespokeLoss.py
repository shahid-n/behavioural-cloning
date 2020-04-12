import keras.backend as kbe

def bespoke_loss(yTrue, yPred):
    sqErr = kbe.square(yPred - yTrue)
    penalty = kbe.exp(kbe.square(yTrue))
    
    return kbe.dot(kbe.transpose(sqErr), penalty)