#include "penhaun_svm.h"

Frame_libsvm::CSvm_wrapper::~CSvm_wrapper()
{
  if (wrapper_model)
  {
    //svm_destroy_model(wrapper_model);
    svm_free_and_destroy_model(&wrapper_model);
    wrapper_model=NULL;
  }
}


Frame_libsvm::CSvm_wrapper::CSvm_wrapper(const std::string &file_name)
{
  wrapper_model=svm_load_model(file_name.c_str());
}

void Frame_libsvm::CSvm_wrapper::set_wrapper_linear()
{
  wrapper_param.kernel_type=LINEAR;
}

void Frame_libsvm::CSvm_wrapper::set_wrapper_rbf(double gamma)
{
  wrapper_param.kernel_type=RBF;
  wrapper_param.gamma=gamma;
}

void Frame_libsvm::CSvm_wrapper::set_wrapper_sigmoid(double gamma, double coef0)
{
  wrapper_param.kernel_type=SIGMOID;
  wrapper_param.gamma=gamma;
  wrapper_param.coef0=coef0;
}

void Frame_libsvm::CSvm_wrapper::set_wrapper_poly(double gamma, double coef0, double degree)
{
  wrapper_param.kernel_type=POLY;
  wrapper_param.gamma=gamma;
  wrapper_param.coef0=coef0;
  wrapper_param.degree=degree;
}

void Frame_libsvm::CSvm_wrapper::set_wrapper_c_svc(double C)
{
  wrapper_param.svm_type=C_SVC;
  wrapper_param.C=C;
}

void Frame_libsvm::CSvm_wrapper::set_wrapper_nu_svc(double nu)
{
  wrapper_param.svm_type=NU_SVC;
  wrapper_param.nu=nu;
}

void Frame_libsvm::CSvm_wrapper::set_wrapper_nu_svr(double nu, double C)
{
  wrapper_param.svm_type=NU_SVR;
  wrapper_param.nu=nu;
  wrapper_param.C=C;
}

void Frame_libsvm::CSvm_wrapper::set_wrapper_epsilon_svr(double eps, double C)
{
  wrapper_param.svm_type=EPSILON_SVR;
  wrapper_param.C=C;
}

void Frame_libsvm::CSvm_wrapper::set_wrapper_one_class(double nu)
{
  wrapper_param.svm_type=ONE_CLASS;
  wrapper_param.nu=nu;
}

void Frame_libsvm::CSvm_wrapper::set_wrapper_cache_size(double mb)
{
  wrapper_param.cache_size=mb;
}

void Frame_libsvm::CSvm_wrapper::set_wrapper_shrinking(bool b)
{
  wrapper_param.shrinking=b?1:0;
}

void Frame_libsvm::CSvm_wrapper::set_wrapper_probability(bool b)
{
  wrapper_param.probability=b?1:0;
}

void Frame_libsvm::CSvm_wrapper::add_wrapper_train_data(double ans, const std::vector<std::pair<int, double> > &vect)
{
  wrapper_data.push_back(make_pair(ans, vect));
}

void Frame_libsvm::CSvm_wrapper::wrapper_train(bool auto_reload)
{
  std::set<int> feats;
  for (size_t i=0; i<wrapper_data.size(); i++)
    for (size_t j=0; j<wrapper_data[i].second.size(); j++)
     feats.insert(wrapper_data[i].second[j].first);

  if (wrapper_param.gamma==-1)
    wrapper_param.gamma=1.0/(std::max(1ul, feats.size()));

  svm_problem prob;

  prob.l = wrapper_data.size();
  prob.y = new double[wrapper_data.size()];
  for (size_t i=0; i<wrapper_data.size(); i++)
    prob.y[i]=wrapper_data[i].first;

  prob.x = new svm_node*[wrapper_data.size()];

  for (size_t i=0; i<wrapper_data.size(); i++)
  {
    prob.x[i]=new svm_node[wrapper_data[i].second.size()+1];
    
    for (size_t j=0; j<wrapper_data[i].second.size(); j++)
    {
      prob.x[i][j].index=wrapper_data[i].second[j].first;
      prob.x[i][j].value=wrapper_data[i].second[j].second;
    }
    
    prob.x[i][wrapper_data[i].second.size()].index=-1;
    prob.x[i][wrapper_data[i].second.size()].value=0;
    
  }

  wrapper_model=svm_train(&prob, &wrapper_param);

  if (auto_reload)
  {
    char tmp[32]="penhaun_svmtempXXXXXX";
    int fd=mkstemp(tmp);
    close(fd);
      
    this->wrapper_save(tmp);
    CSvm_wrapper(tmp).wrapper_swap(*this);
      
    unlink(tmp);
  }

  delete [] prob.y;

  for (int i=0; i<prob.l; i++)
    delete [] prob.x[i];
    
  delete [] prob.x;
}
