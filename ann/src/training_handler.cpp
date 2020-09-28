#include "../include/training_handler.hpp"

CTraining_handler::CTraining_handler()
{
    this->hidden_layer_size = 0;
    this->input_layer_size = 0;
    this->num_classes = 0;
    this->learning_rate = 0;
    this->pos_current_weight = 0;
}

CTraining_handler::~CTraining_handler()
{

}

void CTraining_handler::read_training_csv(std::string path)
{
  std::ifstream data_file;
  data_file.open(path,std::ifstream::in);

  std::string line;
  std::string token;
  std::size_t pos;

  while(std::getline(data_file, line))
  {
    if(line.length() == 0) 
        continue;

    printf("%s \n",line.c_str());

    pos = line.find("training information");
    if (pos != std::string::npos)
        continue;

    pos = line.find("hidden_layer_size");
    if (pos != std::string::npos)
    {
       token = line.substr (pos + std::string("hidden_layer_size").length());
       this->hidden_layer_size = std::stod(token);
       continue;
    }

    pos = line.find("input_layer_size");
    if (pos != std::string::npos)
    {
       token = line.substr (pos + std::string("input_layer_size").length());
       this->input_layer_size = std::stod(token);
       continue;
    }

    pos = line.find("num_classes");
    if (pos != std::string::npos)
    {
       token = line.substr (pos + std::string("num_classes").length());
       this->num_classes = std::stod(token);
       continue;
    }

    pos = line.find("learning_rate");
    if (pos != std::string::npos)
    {
       token = line.substr (pos + std::string("learning_rate").length());
       this->learning_rate = std::stod(token);
       continue;
    }

    pos = line.find("Rows: Output: Delta: Weights:");
    if (pos != std::string::npos)
    {
       continue;
    }

    pos = line.find("Rows:");
    if (pos != std::string::npos)
    {
       token = line.substr (pos + std::string("Rows:").length());
       std::istringstream iss(token);

        do
        {
            std::string sub_string;
            iss >> sub_string;
            if (sub_string == "")
                break;
            neuron_weights.push_back(std::stod(sub_string));
        } while (iss);
       continue;
    }


  }

  data_file.close();
}

int CTraining_handler::get_hiden_layer_size()
{
    return this->hidden_layer_size;
}

int CTraining_handler::get_input_layer_size()
{
    return this->input_layer_size;
}

int CTraining_handler::get_num_classes()
{
    return this->num_classes;
}

double CTraining_handler::get_learning_rate()
{
    return this->learning_rate;
}

double CTraining_handler::next_weight()
{
    double tmp_neuron_weights = (double) 0.0;
    tmp_neuron_weights = this->neuron_weights.at(pos_current_weight);
    pos_current_weight ++;
    return tmp_neuron_weights;
}