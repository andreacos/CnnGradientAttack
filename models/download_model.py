import requests

if __name__ == '__main__':

    url = 'http://clem.dii.unisi.it/~vipp/github/model_keras_ICIP18_64x64x3.h5'
    model_file = 'model_keras_ICIP18_64x64x3.h5'

    print('Downloading model')
    req = requests.get(url, allow_redirects=True)
    open('model_keras_ICIP18_64x64x3.h5', 'wb').write(req.content)
    print('Done!')
