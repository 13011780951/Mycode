//logs.js
const app = getApp()
const util = require('../../utils/util.js')
var md5 = require('/md5.js')
Page({
  data: {
    message0: '小车正在尽力赶过来',
    message1: '请稍等，小车正在尽力赶来',
    message2: '小车正在全力带领你到目的地',
    message3: '小车已经到达目的地',
    message4: '小车已到达出发点，点击出发键即可出发',
  
      //  message1: '小车已到达出发地',
    judge:app.globalData.judge,
    logs: []
  },
  input_startplace:function(e)
  {
    this.setData({
      Name: e.detail.value
    })
    app.globalData.startPlace=this.data.Name
    console.log(app.globalData.startPlace)
  },
  input_endplace:function(e)
  {
    this.setData({
      number: e.detail.value
    })
    app.globalData.endPlace=this.data.number

    console.log(app.globalData.endPlace)
  },
  start:function(e)
  {
  
    
    wx.request({
    url: 'http://192.168.43.239:8887/test/', //仅为示例，并非真实的接口地址
 //   url:'http://127.0.0.1:8000/' ,
    data: {
        'startPlace':app.globalData.startPlace,
        'endPlace':app.globalData.endPlace,
        'signature':md5.hex_md5(app.globalData.startPlace+app.globalData.endPlace),

//        'signature':md5.hex_md5(app.globalData.startPlace,app.globalData.endPlace),
      },

      header: {
        'content-type': 'application/json' // 默认值
      },
      method:'post',
      success: (res) =>{
     
        console.log(res.data)
        this.setData(
          {
            judge:parseInt(res.data.judge)
   //        judge: res.data.message
          },
     
        )
        
      },
        
      fail (res) {
          console.log("fail!!!")
        }
      })
    console.log(app.globalData.endPlace)
  },

goToEndplace:function(e)
{

  wx.request({
    url: 'http://192.168.43.239:8887/gon/', //仅为示例，并非真实的接口地址
    data: {
      'goOrNot':1,
      'startPlace':app.globalData.startPlace,
      'endPlace':app.globalData.endPlace,
      'signature':md5.hex_md5('1'+app.globalData.startPlace+app.globalData.endPlace),

    },
    header: {
      'content-type': 'application/json' // 默认值
    },
    method:'post', 
    
    success: (res)=> {
     
      console.log(res.data),
     
      this.setData(
        {
          judge:parseInt(res.data.judge)
 //        judge: res.data.message
        },
   
      )

    },
      
    fail (res) {
        console.log("fail!!!")
      }
    })

},
ifArrive:function(e)
{

  wx.request({
    url: 'http://192.168.43.239:8887/ifArrive/', //仅为示例，并非真实的接口地址
    data: {
      
    },
    header: {
      'content-type': 'application/json' // 默认值
    },
    method:'post',
    success: (res) =>{
     
      console.log(res.data)
      this.setData(
        {
          judge:parseInt(res.data.judge)
 //        judge: res.data.message
        },
   
      )
      
    },
      
    fail (res) {
        console.log("fail!!!")
      }
    })
 // console.log("轮播请求1秒触发一次")
},

onLoad: function () {
  var that=this
  setInterval(function () {
    that.ifArrive()
 
  }, 3000)    //代表1秒钟发送一次请求
  this.setData({
    logs: (wx.getStorageSync('logs') || []).map(log => {
      return util.formatTime(new Date(log))
    })
  })
}
})