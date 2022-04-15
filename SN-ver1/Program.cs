
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.TimeSeries;
using SN_ver1;



class program
{
    
    static readonly string _trainDataPath = Path.Combine(Environment.CurrentDirectory, "data", "googleSP.csv");
    static readonly string _modelPath = Path.Combine(Environment.CurrentDirectory, "data", "Model.zip");
    static readonly string _testDataPath = Path.Combine(Environment.CurrentDirectory, "data", "googleSp.csv");
    private static MLContext mlContext; 
    static void Main(string[] args)
    {
       
        mlContext = new MLContext(seed: 0);
        IDataView dataView=mlContext.Data.LoadFromTextFile<trainLoad>(path: _trainDataPath,hasHeader: false, separatorChar: ';');
        IDataView dataView2= mlContext.Data.LoadFromTextFile<trainLoadfor2>(path: _trainDataPath, hasHeader: false, separatorChar: ';');
      //   AnomalyDetector(dataView);
       anomalyDetectorSec(dataView2);
        Console.ForegroundColor = ConsoleColor.Magenta;
        Console.WriteLine("===============koniec procesu================");

    }

    static void AnomalyDetector(IDataView dataView)
    {
        Console.ForegroundColor=ConsoleColor.Yellow;
        Console.WriteLine(" rozpoczynam analize zbioru testowego");
        
        string inputColumnName=nameof(trainLoad.Volume);
        string outputColumnName=nameof(outputPrediction.prediction);

        int period = mlContext.AnomalyDetection.DetectSeasonality(dataView, inputColumnName); //z jakiegos powodu zwraca -1 dane nie podlegaja wachaniom sezonowym czyli np z tyg na tydzien nie da sie przez to detectowac spikow jesli zwraca -1
        Console.ForegroundColor = ConsoleColor.White;
        Console.WriteLine("Period of the series is: {0}.",period );
        period += 1;

        var options = new SrCnnEntireAnomalyDetectorOptions()
        {
            Threshold = 0.3,
            Sensitivity = 70.0,
            DetectMode = SrCnnDetectMode.AnomalyAndMargin,
            Period = period,
        };

        var outputDataView = mlContext.AnomalyDetection.DetectEntireAnomalyBySrCnn(dataView, outputColumnName, inputColumnName, options);

        var predictiopns = mlContext.Data.CreateEnumerable<outputPrediction>(
            outputDataView, reuseRowObject: false);
        Console.ForegroundColor = ConsoleColor.Green;
        Console.WriteLine("przewidywane odpowiedzi:");
        var counter = 0;
        Console.WriteLine("Index\tData\tAnomalia\tWartoscAnomali\tMag\toczekiwanaWartosc\tWartoscGraniczna\tGornaGranica\tDolnaGranica");

        foreach(var p in predictiopns)
        {
            if(p.prediction[0]==1)
            {
                Console.ForegroundColor = ConsoleColor.Red;
                Console.WriteLine(" {0} || {1} || {2} || {3} || {4} || {5} || {6} || {7}  ---SPIKE",counter,
                    p.prediction[0],p.prediction[1],p.prediction[2],p.prediction[3],p.prediction[4],p.prediction[5],p.prediction[6]);

            }
            else
            {
                Console.ForegroundColor = ConsoleColor.White;
                Console.WriteLine(" {0} || {1} || {2} || {3} || {4} || {5} || {6} || {7} ", counter,
                    p.prediction[0], p.prediction[1], p.prediction[2], p.prediction[3], p.prediction[4], p.prediction[5], p.prediction[6]);

            }
            counter++;
        }
    }

    static void anomalyDetectorSec(IDataView dataView)
    {
        var pipeline = mlContext.Transforms.DetectSpikeBySsa(nameof(outputPrediction.prediction), nameof(trainLoadfor2.Volume),
            confidence: 98, //do jakiego ulamka od 0 wyznaczac spike dla tego przykladu do 0,2
            trainingWindowSize: 90,
            seasonalityWindowSize: 30,
            pvalueHistoryLength:30
            );
        var tranformedData=pipeline.Fit(dataView).Transform(dataView);

        var predictions = mlContext.Data.CreateEnumerable<outputPrediction>(tranformedData, reuseRowObject: false).ToList();

        var wartosc = dataView.GetColumn<float>("Volume").ToArray();
      
            var date = dataView.GetColumn<DateTime>("data").ToArray();
        
      
      for(int i=0;i<predictions.Count();i++)
        {

            if (predictions[i].prediction[0] == 1)
            {
                Console.ForegroundColor = ConsoleColor.Red;
                Console.WriteLine("{0}||\t{1}\t{2:0.0000}\t{3:0.00}\t{4:0.00}\t{5:0.00} <---SPIKE",
                    i,
                    date[i],
                    wartosc[i],
                    predictions[i].prediction[0],
                    predictions[i].prediction[1],
                    predictions[i].prediction[2]);


            }
            else
            {
                Console.ForegroundColor = ConsoleColor.White;
                Console.WriteLine("{0}||\t{1}\t{2:0.0000}\t{3:0.00}\t{4:0.00}\t{5:0.00}",
                    i,
                    date[i],
                    wartosc[i],
                    predictions[i].prediction[0],
                    predictions[i].prediction[1],
                    predictions[i].prediction[2]);
            }
        }

    }

   
}