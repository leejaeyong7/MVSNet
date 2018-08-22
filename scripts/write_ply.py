''' reads PFM file (from MVSNet output) and writes to PLY file '''

FILE_NAME = 'output.pfm'

def load_pfm(file):
    color = None
    width = None
    height = None
    scale = None
    data_type = None
    header = str(file.readline()).rstrip()

    if header == 'PF':
        color = True
    else:
        raise Exception('Not a Colored PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline())
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')
    scale = float((file.readline()).rstrip())
    if scale < 0: # little-endian
        data_type = '<f'
    else:
        data_type = '>f' # big-endian
    data_string = file.read()
    data = np.fromstring(data_string, data_type)
    shape = (height, width, 3)
    data = np.reshape(data, shape)
    data = cv2.flip(data, 0)
    return data

raw_data = load_pfm(FILE_NAME)
print(raw_data)
