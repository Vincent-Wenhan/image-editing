"""Offline blinded rating form; images and ratings stay on the local filesystem."""

import html

FIELDS = ("task_id", "edit_success", "preservation", "failure", "comment", "reviewer")
FAILURES = (
    "missed_edit",
    "wrong_location",
    "collateral_change",
    "mask_artifact",
    "geometry",
    "other",
    "unjudgeable",
)


def page(tasks: list[dict]) -> str:
    cards = []
    scores = '<option value=""></option>' + "".join(
        f"<option>{value}</option>" for value in ("0", "1", "2", "NA")
    )
    failures = '<option value=""></option>' + "".join(
        f"<option>{value}</option>" for value in FAILURES
    )
    for task in tasks:
        pictures = "".join(
            f'<figure><img src="images/{task["task_id"]}_{key}.png">'
            f"<figcaption>{key}</figcaption></figure>"
            for key in ("source", "target", "prediction")
        )
        cards.append(
            f'<article data-task="{task["task_id"]}"><h2>{task["task_id"]}</h2>'
            f'<p>{html.escape(task["instruction"])}</p><div class="images">{pictures}</div>'
            f'<label>编辑成功度 <select name="edit_success">{scores}</select></label> '
            f'<label>无关内容保留 <select name="preservation">{scores}</select></label> '
            f'<label>失败类别 <select name="failure">{failures}</select></label>'
            '<label>说明 <input name="comment" size="60"></label></article>'
        )
    return (
        """<!doctype html><meta charset="utf-8"><title>Blinded editing review</title>
<style>body{font:16px system-ui;margin:24px;background:#eee}article{background:white;
padding:16px;margin:16px 0}.images{display:flex;flex-wrap:wrap}figure{margin:4px}
img{width:300px;max-width:100%}h2{font-size:16px}label{display:inline-block;margin:8px}</style>
<h1>编辑盲评</h1><p>请先填写评审者代号，再独立评价每个输出。target 仅作编辑意图参考，
不要求生成与 target 像素相等。请勿查看 private/ 中的方法映射。</p>
<p>编辑成功度：0=未完成或错误；1=部分完成；2=完整且位置/属性正确。
无关内容保留：0=明显破坏；1=有可见偏移；2=良好保留（忽略必要的物理后果）。
无法判断选 NA，并选 unjudgeable、说明原因。评分低于2时填写失败类别及说明。
未评分留空。下载 CSV 后使用 review-score 导入；本页不联网、不上传。</p>
<label>评审者 <input id="reviewer"></label><button id="export">下载评分 CSV</button>
"""
        + "\n".join(cards)
        + r"""
<script>
const fields=['task_id','edit_success','preservation','failure','comment','reviewer'];
const quote=v=> '"'+String(v).replaceAll('"','""')+'"';
document.querySelector('#export').onclick=()=>{
 const reviewer=document.querySelector('#reviewer').value;
 const rows=[fields,...[...document.querySelectorAll('article')].map(card=>
 fields.map(key=>key==='task_id'?card.dataset.task:key==='reviewer'?reviewer:
 card.querySelector('[name="'+key+'"]').value))];
 const a=document.createElement('a');
 a.href=URL.createObjectURL(new Blob(['\ufeff'+rows.map(row=>row.map(quote).join(',')).join('\n')],
 {type:'text/csv;charset=utf-8'}));a.download='ratings.csv';a.click();URL.revokeObjectURL(a.href);
};
</script>"""
    )
