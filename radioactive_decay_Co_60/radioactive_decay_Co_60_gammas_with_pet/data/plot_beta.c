void plot_beta()
{
   TCanvas * c1 = new TCanvas();
   TFile *file = new TFile("Canvas_beta.root","read");
  
   TH1F *hist = (TH1F*)file->Get("tcanvas1");
   
   hist->Draw();   
   file->Close();
}
