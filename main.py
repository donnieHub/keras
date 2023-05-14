import MnistDigitsRecognition
from NeuronNetwork import NeuronNetwork


def print_hi(name):
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
    #NeuronNetwork().create_neuron_network()
    MnistDigitsRecognition.Mnist().test_mnist()
