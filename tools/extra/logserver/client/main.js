/* global $, Chartist */
'use strict';
$(function(){
  plot();
})

function plotTraining(data) {
  let points = Math.floor(data.labels.length / 20);
  new Chartist.Line('.ct-chart-training-accuracy', 
    {labels: data.labels, 
     series: data.series.slice(0,2)
    }, {
    axisX: {
      labelInterpolationFnc: function(value, index) {
        return index % points === 0 ? value : null;
      }
    }
  });
  new Chartist.Line('.ct-chart-training-loss', 
    {labels: data.labels,
     series: data.series.slice(2)
    }, {
    axisX: {
      labelInterpolationFnc: function(value, index) {
        return index % points === 0 ? value : null;
      }
    }
  });
}

function plotTest(data) {
  let points = Math.floor(data.labels.length / 20);
  new Chartist.Line('.ct-chart-test-accuracy', 
    {labels: data.labels, 
     series: data.series.slice(0,1)
    }, {
    axisX: {
      labelInterpolationFnc: function(value, index) {
        return index % points === 0 ? value : null;
      }
    }
  });
  new Chartist.Line('.ct-chart-test-loss', 
    {labels: data.labels,
     series: data.series.slice(1,2)
    }, {
    axisX: {
      labelInterpolationFnc: function(value, index) {
        return index % points === 0 ? value : null;
      }
    }
  });
}

function plot() {
  $('.ct-chart').empty();
  $.get('/data')
    .done(function(res){
      var data = res.data;
      plotTraining(data[0]);
      plotTest(data[1]);
    })
    .fail(function(err) {
      console.error(err);
    });
}

$('.plot-btn').click(function() {
  plot();
});

$('#submitLogFile').click( function() {
  $.post('/logfile', $('form#logFileForm').serialize(), 
     function(data) {
       plot();
     },
     'json');
});
