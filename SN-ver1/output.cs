using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SN_ver1
{
     class outputPrediction
    {
        [VectorType(7)]
        public double[] prediction { get; set; }
    }
}
