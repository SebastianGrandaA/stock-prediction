module Constants

INPUT_PATH = "input"
INPUT_DATA_PATH = "$(INPUT_PATH)/data"
DEFAULT_SETTINGS_FILE = "$(INPUT_PATH)/settings.json"
DEFAULT_DATA_SPLIT = ["train", "test", "validation"]
DEFAULT_INDIVIDUAL_FILES = [
    "$(INPUT_DATA_PATH)/individual/144char_1976-1995_train.npz",
    "$(INPUT_DATA_PATH)/individual/144char_2001-2020_test.npz",
    "$(INPUT_DATA_PATH)/individual/144char_1996-2000_validation.npz",
]
DEFAULT_MACROECONOMIC_FILES = [
    "$(INPUT_DATA_PATH)/macroeconomic/285mc_1976-1995_train.npz",
    "$(INPUT_DATA_PATH)/macroeconomic/285mc_2001-2020_test.npz",
    "$(INPUT_DATA_PATH)/macroeconomic/285mc_1996-2000_validation.npz",
]

DEFAULT_OUTPUT_PATH = "output"

BASE_TRAIN_INFO = (
    epoch=Int64[],
    discriminator_loss=Float64[],
    generator_loss=Float64[],
    fixed_discriminator_loss=Float64[],
)
BASE_METRICS = (ID=String[], predictions=Array[], sharpe_ratio=Float64[])

DEFAULT_EPOCHS = 100
DEFAULT_SUB_EPOCHS = 5
DEFAULT_LOG_EPOCH = 1

UNKNOWN = -99.99

DEFAULT_DIMS = [10, 10]
DEFAULT_OUTPUT_DIM = 1

end # module Constants