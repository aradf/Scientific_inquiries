void plot_photon()
{
   TCanvas * c1 = new TCanvas();
   TFile *file = new TFile("Canvas_photon.root","READ");

   TH1F *hist = (TH1F*)file->Get("tcanvas1");
   
   hist->Draw();   
   file->Close();
}
