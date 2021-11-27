from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv2D, BatchNormalization, LeakyReLU, MaxPool2D, UpSampling2D, concatenate, add



def create_downsample (channels, inputs):
    x = BatchNormalization (momentum = 0.9)(inputs)
    x = LeakyReLU (0)(x)
    x = Conv2D (channels, 1, padding = 'same', use_bias = False)(x)
    x = MaxPool2D (2)(x)
    # Added start
    #x = Conv2D (channels, 1, padding = 'same', use_bias = False)(x)
    #x = MaxPool2D (2)(x)
    # Added End
    return x


def create_resblock (channels, inputs):
    x = BatchNormalization (momentum = 0.9)(inputs)
    x = LeakyReLU (0)(x)
    x = Conv2D (channels, 3, padding='same', use_bias = False)(x)
    x = BatchNormalization (momentum = 0.9)(x)
    x = LeakyReLU (0)(x)
    x = Conv2D (channels, 3, padding = 'same', use_bias = False)(x)

    #Added Start
    x = BatchNormalization (momentum = 0.9)(x)
    x = LeakyReLU (0)(x)
    x = Conv2D (channels, 3, padding = 'same', use_bias = False)(x)
    #Added End
    
    addInput = x;
    print ("Add input shape:", addInput.shape)
    print ("Resnet block input shape:", inputs.shape)
    resBlockOut = add ([addInput, inputs])
    print ("Resnet block out shape:", resBlockOut.shape)
    out = concatenate([resBlockOut, addInput], axis = 3)
    print ("concat block out shape:", out.shape)
    out = Conv2D (channels, 1, padding = 'same', use_bias = False)(out)
    print ("mixed block out shape:", out.shape)
    return out

def create_network (input_size, channels, n_blocks = 2, depth = 4):
    # input
    inputs = Input (shape = (input_size, input_size, 1))
    x = Conv2D (channels, 3, padding = 'same', use_bias = False)(inputs)
    # residual blocks
    for d in range (depth):
        channels = channels * 2
        x = create_downsample (channels, x)
        for b in range (n_blocks):
            x = create_resblock (channels, x)
    # output
    x = BatchNormalization (momentum = 0.9)(x)
    x = LeakyReLU (0)(x)
    x = Conv2D (1, 1, activation = 'sigmoid')(x)
    outputs = UpSampling2D (2**depth)(x)
    model = Model (inputs = inputs, outputs = outputs)
    return model