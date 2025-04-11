#include <iostream>
#include <vector>
#include <random>
#include <cmath>

double random()
{
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_real_distribution<double> dist(-1.0, 1.0); 

    return dist(gen);
}

double sigmoid(const double& x)
{
    return (1.0 / (1.0 + std::exp(-x)));
}

double sigmoid_derivative(const double& x)
{
    return (x * (1.0 - x));
}

namespace AI
{
    class NeuralNetwork
    {
    public:
        NeuralNetwork(const int& input_size, const int& hidden1_size, const int& output_size);
        
        std::vector<double> FeedForward(const std::vector<double>& input);
        void Train(const std::vector<double>& input, const std::vector<double>& target, const double& learn_rate = 0.1);
        
    private:
        int m_input_size, m_hidden1_size, m_output_size;
        
        std::vector<double> m_hidden1_neurons;

        std::vector<double> m_hidden1_bias;
        std::vector<double> m_output_bias;
        
        std::vector<std::vector<double>> m_inputs_weights;
        std::vector<std::vector<double>> m_hidden1_weights;
        
        void Init();
        
    };

    NeuralNetwork::NeuralNetwork(
        const int& input_size, const int& hidden1_size, const int& output_size) :
        m_input_size(input_size), m_hidden1_size(hidden1_size), m_output_size(output_size)
    {
        Init();
    }
    
    void NeuralNetwork::Init()
    {
        // Input Layer
        m_inputs_weights.resize(m_input_size, std::vector<double>(m_hidden1_size, 0.0));
    
        for (int i = 0; i < m_input_size; i++)
            for (int j = 0; j < m_hidden1_size; j++)
                m_inputs_weights[i][j] = random();
    
        // Hidden 1 Layer
        m_hidden1_neurons.resize(m_hidden1_size, 0.0);
        m_hidden1_bias.resize(m_hidden1_size, 0.0);
        m_hidden1_weights.resize(m_hidden1_size, std::vector<double>(m_output_size, 0.0));
    
        for (int i = 0; i < m_hidden1_size; i++)
            m_hidden1_bias[i] = random();
    
        for (int i = 0; i < m_hidden1_size; i++)
            for (int j = 0; j < m_output_size; j++)
                m_hidden1_weights[i][j] = random();
    
        // Output Layer
        m_output_bias.resize(m_output_size, 0.0);
    
        for (int i = 0; i < m_output_size; i++)
            m_output_bias[i] = random();
    }
    
    std::vector<double> NeuralNetwork::FeedForward(const std::vector<double>& input)
    {
        std::vector<double> output(m_output_size, 0.0);
    
        for (int i = 0; i < m_hidden1_size; i++)
        {
            m_hidden1_neurons[i] = 0.0;
            for (int j = 0; j < m_input_size; j++)
                m_hidden1_neurons[i] += input[j] * m_inputs_weights[j][i];
            m_hidden1_neurons[i] = sigmoid(m_hidden1_neurons[i] +  m_hidden1_bias[i]);
        }
    
        for (int i = 0; i < m_output_size; i++)
        {
            output[i] = 0.0;
            for (int j = 0; j < m_hidden1_size; j++)
                output[i] += m_hidden1_neurons[j] * m_hidden1_weights[j][i];
            output[i] = sigmoid(output[i] + m_output_bias[i]);
        }
            
        return output;
    }
    
    void NeuralNetwork::Train(const std::vector<double>& input, const std::vector<double>& target, const double& learn_rate)
    {
        std::vector<double> output = FeedForward(input);
    
        // Calc Errors & Delats
        std::vector<double> output_errors(m_output_size, 0.0);
        for (int i = 0; i < m_output_size; i++)
            output_errors[i] = target[i] - output[i];
    
        std::vector<double> output_deltas(m_output_size, 0.0);
        for (int i = 0; i < m_output_size; i++)
            output_deltas[i] = output_errors[i] * sigmoid_derivative(output[i]);
    
        std::vector<double> hidden1_errors(m_hidden1_size, 0.0);
        for (int i = 0; i < m_hidden1_size; i++)
            for (int j = 0; j < m_output_size; j++)
                hidden1_errors[i] += output_deltas[j] * m_hidden1_weights[i][j];
    
        std::vector<double> hidden1_deltas(m_hidden1_size, 0.0);
        for (int i = 0; i < m_hidden1_size; i++)
            hidden1_deltas[i] = hidden1_errors[i] * sigmoid_derivative(m_hidden1_neurons[i]);
    
        // Update Weights
        for (int i = 0; i < m_input_size; i++)
            for (int j = 0; j < m_hidden1_size; j++)
                m_inputs_weights[i][j] += learn_rate * hidden1_deltas[j] * input[i];
    
        for (int i = 0; i < m_hidden1_size; i++)
            for (int j = 0; j < m_output_size; j++)
                m_hidden1_weights[i][j] += learn_rate * output_deltas[j] * m_hidden1_neurons[i];
    
        // Update Bais
        for (int i = 0; i < m_hidden1_size; i++)
            m_hidden1_bias[i] += learn_rate * hidden1_deltas[i];
    
        for (int i = 0; i < m_output_size; i++)
            m_output_bias[i] += learn_rate * output_deltas[i];
    }
}



int main() {
    AI::NeuralNetwork nn(2, 4, 1);

    std::vector<std::vector<double>> inputs = {
        {0.0, 0.0},
        {0.0, 1.0},
        {1.0, 0.0},
        {1.0, 1.0}
    };

    std::vector<std::vector<double>> outputs = {
        {0.0},
        {1.0},
        {1.0},
        {0.0}
    };

    double learn_rate = 2.0;
    for (int epoch = 0; epoch < 10000; epoch++)
    {
        for (int i = 0; i < inputs.size(); i++)
            nn.Train(inputs[i], outputs[i], learn_rate);

        if (epoch%1000 == 0)
            learn_rate *= 0.9;
    }
        
    
    for (int i = 0; i < inputs.size(); i++)
    {
        auto output = nn.FeedForward(inputs[i]);
        std::cout << "[" << inputs[i][0] << ", " << inputs[i][1] << "] => [" << output[0] << "]\n";
    }
    
    return 0;
}
