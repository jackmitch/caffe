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
  new Chartist.Line('.ct-chart-test', 
    {labels: data.labels,
     series: data.series.slice(0,2)
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
  $.get('/api')
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
