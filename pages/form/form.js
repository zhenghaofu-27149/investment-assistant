Page({
  data: {
    target: "",      // 投资标的，比如 PDD / BTC
    reason: "",      // 研究&逻辑
    positionPlan: "" // 仓位计划
  },

  // 绑定输入：标的
  onInputTarget(e) {
    this.setData({
      target: e.detail.value
    });
  },

  // 绑定输入：研究逻辑
  onInputReason(e) {
    this.setData({
      reason: e.detail.value
    });
  },

  // 绑定输入：仓位计划
  onInputPosition(e) {
    this.setData({
      positionPlan: e.detail.value
    });
  },

  // 提交分析
  submit() {
    const { target, reason, positionPlan } = this.data;

    if (!target || !reason) {
      wx.showToast({
        title: "标的和研究理由必填",
        icon: "none"
      });
      return;
    }

    wx.showLoading({
      title: "分析中…",
      mask: true
    });

    wx.request({
      url: "http://127.0.0.1:8000/evaluate", // 本地 FastAPI
      method: "POST",
      header: {
        "content-type": "application/json"
      },
      data: {
        asset_name: target,
        research_text: reason,        // 这里简单用 reason，当成研究说明
        thesis_text: reason,          // 你后面要的话可以拆成两个字段
        plan_text: positionPlan,      // 仓位/止损计划
        emotion_score: 4,             // 先写死，后面再加一个情绪滑条输入
        capital_impact: "",
        position_pct: positionPlan    // 先直接用文本，你后面可以规范成数字
      },
      success: (res) => {
        wx.hideLoading();
        console.log("后端返回：", res.data);

        // 简单暴力：直接弹窗显示整体评分+建议
        const data = res.data;
        const msg = `
总分：${data.overall_score}（${data.verdict}）

重点问题：
${(data.flags || []).join("；") || "无"}

建议：
${data.advice || ""}
        `.trim();

        wx.showModal({
          title: "分析结果",
          content: msg,
          confirmText: "我知道了",
          showCancel: false
        });
      },
      fail: (err) => {
        wx.hideLoading();
        console.error("请求失败：", err);
        wx.showToast({
          title: "请求失败，检查后端服务",
          icon: "none"
        });
      }
    });
  }
});
