#include "../include/data_handler.hpp"
#include <algorithm>
#include <random>
#include "../include/helper_funtion.hpp"

CData_handler::CData_handler()
{
  debug_print("CData_handler::CData_handler()");
  data_array = new std::vector<CData *>;
  training_data = new std::vector<CData *>;
  test_data = new std::vector<CData *>;
  validation_data = new std::vector<CData *>;
  output_weight_file_handler = new std::ofstream();
}

CData_handler::~CData_handler()
{
  debug_print("CData_handler::~CData_handler()");
  if (data_array != nullptr)
    delete data_array;

  if (training_data != nullptr)
    delete training_data;

  if (test_data != nullptr)
    delete test_data;

  if (validation_data != nullptr)
    delete validation_data;

  if (output_weight_file_handler != nullptr)
  {
    output_weight_file_handler->close();
    delete output_weight_file_handler;
  }
  
}

void CData_handler::read_csv(std::string path, std::string delimiter)
{
  classification_counts = 0;
  std::ifstream data_file;
  //data_file.open(path.c_str(),std::ifstream::in);
  data_file.open(path,std::ifstream::in);

  std::string line;

  while(std::getline(data_file, line))
  {
    if(line.length() == 0) continue;
    CData *data = new CData();
    data->set_normalized_feature_vector(new std::vector<double>());

    size_t position = 0;
    std::string token;
    while((position = line.find(delimiter)) != std::string::npos)
    {
      token = line.substr(0, position);
      data->append_to_feature_vector(std::stod(token));
      line.erase(0, position + delimiter.length());
    }

    if(class_from_string.find(line) != class_from_string.end())
    {
      data->set_label(class_from_string[line]);
    } 
    else 
        {
          class_from_string[line] = classification_counts;
          data->set_label(class_from_string[token]);
          classification_counts++;
        }

    data_array->push_back(data);
  }

  for(CData *data_item : *data_array)
    data_item->set_class_vector(classification_counts);
  //normalize();
  feature_vector_size = data_array->at(0)->get_normalized_feature_vector()->size();
  data_file.close();
}

void CData_handler::read_input_data(std::string path)
{
  uint32_t magic = 0;
  uint32_t num_images = 0;
  uint32_t num_rows = 0;
  uint32_t num_cols = 0;

  unsigned char bytes[4];
  FILE *f = fopen(path.c_str(), "r");
  if(f)
  {
    int i = 0;
    while(i < 4)
    {
      if(fread(bytes, sizeof(bytes), 1, f))
      {
        switch(i)
        {
          case 0:
            magic = format(bytes);
            i++;
            break;
          case 1:
            num_images = format(bytes);
            i++;
            break;
          case 2:
            num_rows = format(bytes);
            i++;
            break;
          case 3:
            num_cols = format(bytes);
            i++;
            break;
        }
      }
    }
    printf("Done getting file header.\n");
    uint32_t image_size = num_rows * num_cols;
    for(i = 0; i < num_images; i++)
    {
      CData *some_data = new CData();
      some_data->set_feature_vector(new std::vector<uint8_t>());

      uint8_t element[1];
      for(int j = 0; j < image_size; j++)
      {
        if(fread(element, sizeof(element), 1, f))
        {
          some_data->append_to_feature_vector(element[0]);
        }
      }
      data_array->push_back(some_data);
      data_array->back()->set_class_vector(classification_counts);
    }
    normalize();
    feature_vector_size = data_array->at(0)->get_feature_vector()->size();
    printf("Successfully read %lu data entries.\n", data_array->size());
    printf("The Feature Vector Size is: %d \n", feature_vector_size);
  } else
  {
    printf("Invalid Input File Path");
    exit(1);
  }
}

void CData_handler::open_weight_data_file(std::string path)
{
  output_weight_file_handler->open(path);
}

void CData_handler::write_weight_data(std::string weights)
{
  
  if (output_weight_file_handler != nullptr)
  {
    *output_weight_file_handler << weights;
  }

}

void CData_handler::read_label_data(std::string path)
{
  uint32_t magic = 0;
  uint32_t num_images = 0;
  unsigned char bytes[4];
  FILE *f = fopen(path.c_str(), "r");
  if(f)
  {
    int i = 0;
    while(i < 2)
    {
      if(fread(bytes, sizeof(bytes), 1, f))
      {
        switch(i)
        {
          case 0:
            magic = format(bytes);
            i++;
            break;
          case 1:
            num_images = format(bytes);
            i++;
            break;
        }
      }
    }

    for(unsigned j = 0; j < num_images; j++)
    {
      uint8_t element[1];
      if(fread(element, sizeof(element), 1, f))
      {
        data_array->at(j)->set_label(element[0]);
      }
    }

    printf("Done getting Label header.");
  } 
  else
  {
    printf("Invalid Label File Path");
    exit(1);
  }
}
void CData_handler::split_data()
{
  std::unordered_set<int> used_indexes;
  int train_size = data_array->size() * TRAIN_SET_PERCENT;
  int test_size  = data_array->size() * TEST_SET_PERCENT;
  int valid_size = data_array->size() * VALID_SET_PERCENT;
  
  std::random_shuffle(data_array->begin(), data_array->end());

  // Training Data

  int count = 0;
  int index = 0;
  while(count < train_size)
  {
    training_data->push_back(data_array->at(index++));
    count++;
  }

  // Test Data
  count = 0;
  while(count < test_size)
  {
    test_data->push_back(data_array->at(index++));
    count++;
  }

  // Test Data

  count = 0;
  while(count < valid_size)
  {
    validation_data->push_back(data_array->at(index++));
    count++;
  }

  printf("Training Data Size: %lu.\n", training_data->size());
  printf("Test Data Size: %lu.\n", test_data->size());
  printf("Validation Data Size: %lu.\n", validation_data->size());
}

void CData_handler::count_classes()
{
  int count = 0;
  for(unsigned i = 0; i < data_array->size(); i++)
  {
    if(class_from_int.find(data_array->at(i)->get_label()) == class_from_int.end())
    {
      class_from_int[data_array->at(i)->get_label()] = count;
      data_array->at(i)->set_enumerated_label(count);
      count++;
    }
    else 
    {
      data_array->at(i)->set_enumerated_label(class_from_int[data_array->at(i)->get_label()]);
    }
  }
  
  classification_counts = count;
  for(CData *data : *data_array)
    data->set_class_vector(classification_counts);
  printf("Successfully Extraced %d Unique Classes.\n", classification_counts);
}

void CData_handler::normalize()
{
  std::vector<double> mins, maxs;
  // fill min and max lists
  
  CData *d = data_array->at(0);
  for(auto val : *d->get_feature_vector())
  {
    mins.push_back(val);
    maxs.push_back(val);
  }

  for(int i = 1; i < data_array->size(); i++)
  {
    d = data_array->at(i);
    for(int j = 0; j < d->get_feature_vector_size(); j++)
    {
      //double value = (double) d->getFeatureVector()->at(j);
      double value = (double) d->get_feature_vector_size();
      if(value < mins.at(j)) mins[j] = value;
      if(value > maxs.at(j)) maxs[j] = value;
    }
  }
  // normalize data array
  
  for(int i = 0; i < data_array->size(); i++)
  {
    data_array->at(i)->set_normalized_feature_vector(new std::vector<double>());
    data_array->at(i)->set_class_vector(classification_counts);
    for(int j = 0; j < data_array->at(i)->get_feature_vector_size(); j++)
    {
      if(maxs[j] - mins[j] == 0) data_array->at(i)->append_to_feature_vector(0.0);
      else
        data_array->at(i)->append_to_feature_vector(
          (double)(data_array->at(i)->get_feature_vector()->at(j) - mins[j])/(maxs[j]-mins[j]));
    }
  }
}

int CData_handler::get_classification_counts()
{
  return classification_counts;
}

int CData_handler::get_data_array_size()
{
  return data_array->size();
}
int CData_handler::get_training_data_size()
{
  return training_data->size();
}
int CData_handler::get_test_data_size()
{
  return test_data->size();
}
int CData_handler::get_validation_size()
{
  return validation_data->size();
}

uint32_t CData_handler::format(const unsigned char* bytes)
{
  return (uint32_t)((bytes[0] << 24) |
                    (bytes[1] << 16)  |
                    (bytes[2] << 8)   |
                    (bytes[3]));
}

std::vector<CData *> * CData_handler::get_data_array()
{
  return data_array;
}

std::vector<CData *> * CData_handler::get_training_data()
{
  return training_data;
}
std::vector<CData *> * CData_handler::get_test_data()
{
  return test_data;
}
std::vector<CData *> * CData_handler::get_validation_data()
{
  return validation_data;
}

std::map<uint8_t, int> CData_handler::get_class_map()
{
  return class_from_int;
}

void CData_handler::print()
{
  printf("Training Data:\n");
  for(auto data : *training_data)
  {
    for(auto value : *data->get_normalized_feature_vector())
    {
      printf("%.3f,", value);
    }
    printf(" ->   %d\n", data->get_label());
  }
  return;

  printf("Test Data:\n");
  for(auto data : *test_data)
  {
    for(auto value : *data->get_normalized_feature_vector())
    {
      printf("%.3f,", value);
    }
    printf(" ->   %d\n", data->get_label());
  }

  printf("Validation Data:\n");
  for(auto data : *validation_data)
  {
    for(auto value : *data->get_normalized_feature_vector())
    {
      printf("%.3f,", value);
    }
    printf(" ->   %d\n", data->get_label());
  }

}

