using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SN_ver1
{
     class trainLoadfor2
    {

        [LoadColumn(0)]
    
        public DateTime data { get; set; }

        [LoadColumn(6)]
    
        public float Volume { get; set; }


    }
}
