'use strict';

let port = process.env.PORT || 4000;
let client = 'client';
let path = require('path');
let fs = require('fs');
let parse = require('csv-parse');
let split = require('split');
let express = require('express');
let app = express();
let pythonShell = require('python-shell');

app.use(express.static(path.join(__dirname, client)));

let logfile_path = "/tmp/caffe.INFO";
let python_parser_path = '../parse_log.py';

var readLogFile = function(logfile_path) {
  return new Promise(function (resolve, reject) {
    let stream = fs.createReadStream(path.join(__dirname, logfile_path)).pipe(split());
    let data = {
      labels: [],
      series: [[], []]
    };
    let linecnt = 0;
    let isTest = false;
    stream.on('data', (row) => {
      if (!row){
        return;
      }
      let tuple = row.split(',');

      if(linecnt == 0) {
        for(let i=0; i < tuple.length; i++) {
          if(tuple[i] == 'accuracy' || tuple[i] == 'acc/top-1') {
            isTest = true;
          }
        }
        linecnt++;
      } else {
        data.labels.push(parseFloat(tuple[0]));
        if(isTest) {
          data.series[0].push(parseFloat(tuple[3]));
          data.series[1].push(parseFloat(tuple[4]));
        } else {
          data.series[0].push(parseFloat(tuple[3]));
        }
      }
    });

    stream.on('end', () => {
      resolve(data);
      console.timeEnd('/api');
    });
  })
  
}

app.get('/api', (req, res, next) => {
  console.time('/api');

  // parse the current log file using the python script
  var options = {
    mode: 'text',
    args: [logfile_path, __dirname]
  };
 
  pythonShell.run(python_parser_path, options, function (err, results) {
    console.log('Log file parsed ' + err);

    if (err) throw err;
    
    // now read the csv files for the train
    Promise.all([
      readLogFile(path.basename(logfile_path) + '.train'),
      readLogFile(path.basename(logfile_path) + '.test')
    ]).then(function(results) {
       res.json({message: 'success', data: results});
    })
  });
});

app.listen(port, () => {
  console.log(`Server listenting on port: ${port}`);
})
