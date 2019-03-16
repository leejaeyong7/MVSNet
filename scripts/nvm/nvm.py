from nvm_model import NVMModel
class NVM:
    '''Parser for NVM file
    
    High level parser that reads NVM file and parses Multiple NVM Models
    (Currently, it only supports single model :( )
    
    NVM format reference: 
      http://ccwu.me/vsfm/doc.html#nvm 
    '''
    def __init__(self):
        self.models = []

    def from_file(self, file_path, image_path):
        '''Reads from NVM file to create multiple models'''
        with open(file_path, 'r') as nvm_file:
            # check NVM header
            header = nvm_file.readline();
            if('NVM_V3' not in header):
                raise Exception('Invalid NVM File')

            # TODO: handle multi-model loading
            model = NVMModel()
            model.from_file(nvm_file, image_path)
            self.models.append(model)
        return self

