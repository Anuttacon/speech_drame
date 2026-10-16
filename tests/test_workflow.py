import argparse
import ast
import importlib.util
import json
from pathlib import Path
import struct
import subprocess
import sys
from types import SimpleNamespace
import wave

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('workflow', ROOT / 'run.py')
w = importlib.util.module_from_spec(spec)
spec.loader.exec_module(w)


def cli(*args):
    return subprocess.run([sys.executable, str(ROOT / 'run.py'), *map(str,args)], cwd=ROOT,
                          capture_output=True, text=True)


def wav(path):
    with wave.open(str(path),'wb') as f:
        f.setparams((1,2,16000,0,'NONE','not compressed'))
        f.writeframes(struct.pack('<100h', *range(100)))


def dataset(tmp_path, track='realism', split='test', count=4):
    source=tmp_path / (track+'-'+split)
    source.mkdir()
    records=[]
    for i in range(count):
        key=f'{split}_{i}'
        wav(source/(key+'.wav'))
        labels=list(w.LABELS.values()) if track=='realism' else w.DIMENSIONS[track]
        annotations={'raters':dict.fromkeys(labels,[i+1]*3), **dict.fromkeys(labels,i+1)}
        annotations['content_pass']=True
        records.append({'id':key,'perspective':track,'split':split,'language':'en',
                        'profile':'Profile','scene':{'local':'Scene'},'char_style':'calm',
                        'transcript':'Words','prompt':'Role question','annotations':annotations,
                        'audio':{'synthesized':{'path':key+'.wav'}}})
    meta=source/'metadata.jsonl';w.write_rows(meta,records)
    return source,meta,records


def prep(tmp_path, source, meta, track='realism', **kwargs):
    args=argparse.Namespace(metadata=[meta],asset_root=source,track=track,purpose='evaluate',
                           protocol='benchmark',ids=None,train_ids=None,test_ids=None,audio_map=None,
                           score_only=False,output=tmp_path/'prepared')
    for key,value in kwargs.items():setattr(args,key,value)
    return w.prepare(args)


def infer_args(work, **kwargs):
    args=argparse.Namespace(workdir=work,backend='qwen2audio',name='test',model=None,
          checkpoint=None,base_model=None,examples=None,num_examples=3,trials=1,
          batch_size=4,zero_audio=False,zero_text=False,dry_run=True)
    for key,value in kwargs.items():setattr(args,key,value)
    return args


@pytest.mark.parametrize('track',['realism','archetype'])
def test_canonical_to_model_mapping_and_canonical_labels_untouched(tmp_path,track):
    source,meta,records=dataset(tmp_path,track)
    out=prep(tmp_path,source,meta,track)
    inputs=w.rows(out/'inputs.jsonl')
    assert w.rows(out/'metadata.jsonl')==records
    assert all(Path(r['wav_path']).is_file() for r in inputs)
    assert inputs[0]['dataset_name']==('ROLE' if track=='realism' else 'Arche')
    if track=='realism':
        assert inputs[0]['char_profile']=='Profile'
        assert inputs[0]['local_scene']=='Scene'
        assert inputs[0]['annotations']['stress_and_emphasis']==[1,1,1]
    else:
        assert inputs[0]['question']=='Role question'
        assert inputs[0]['annotations']['content_pass']==[True]


def test_flac_becomes_actual_pcm_wav(tmp_path):
    import soundfile as sf
    source,meta,records=dataset(tmp_path,count=1)
    flac=source/'audio.flac'
    sf.write(flac,[[.1],[.2],[.3]],16000,format='FLAC')
    records[0]['audio']['synthesized']['path']='audio.flac'
    w.write_rows(meta,records)
    out=prep(tmp_path,source,meta)
    with wave.open(w.rows(out/'inputs.jsonl')[0]['wav_path']) as f:
        assert f.getnframes()==3
        assert f.getsampwidth()==2


def test_mel_only_has_actionable_path_and_score_only_works(tmp_path):
    source,meta,records=dataset(tmp_path)
    for r in records:r['audio']={'spectrogram':{'path':'mel.npz'}}
    w.write_rows(meta,records)
    with pytest.raises(ValueError,match='Mel-only'):
        prep(tmp_path,source,meta)
    out=prep(tmp_path,source,meta,score_only=True)
    with pytest.raises(ValueError,match='Score-only'):
        w.infer_plan(infer_args(out))


def test_external_reconstruction_map(tmp_path):
    source,meta,records=dataset(tmp_path,count=1)
    records[0]['audio']={};w.write_rows(meta,records)
    mapping=source/'map.jsonl';w.write_rows(mapping,[{'id':'test_0','wav_path':'test_0.wav'}])
    out=prep(tmp_path,source,meta,audio_map=mapping)
    assert w.bundle(out)['audio_ready']


def test_authentic_filters_pool_and_retains_disjoint_manifests(tmp_path):
    source,meta,records=dataset(tmp_path,split='test_realrec')
    train=source/'train.jsonl';test=source/'test.jsonl'
    w.write_rows(train,records[:1]);w.write_rows(test,records[1:])
    out=prep(tmp_path,source,meta,protocol='authentic',train_ids=train,test_ids=test)
    assert len(w.rows(out/'inputs.jsonl'))==3
    assert {w.rid(r) for r in w.rows(out/'test_ids.jsonl')}=={r['id'] for r in records[1:]}


def test_authentic_overlap_and_missing_manifests_fail_before_output(tmp_path):
    source,meta,records=dataset(tmp_path)
    with pytest.raises(ValueError,match='requires'):
        prep(tmp_path,source,meta,protocol='authentic')
    ids=source/'ids.jsonl';w.write_rows(ids,records)
    with pytest.raises(ValueError,match='overlap'):
        prep(tmp_path,source,meta,protocol='authentic',train_ids=ids,test_ids=ids)
    assert not (tmp_path/'prepared').exists()


def test_training_can_select_adaptation_only_with_membership_evidence(tmp_path):
    source,meta,records=dataset(tmp_path,split='test_realrec')
    with pytest.raises(ValueError,match='Training requires'):
        prep(tmp_path,source,meta,purpose='train')
    train=source/'train.jsonl';test=source/'test.jsonl'
    w.write_rows(train,records[:1]);w.write_rows(test,records[1:])
    out=prep(tmp_path,source,meta,purpose='train',train_ids=train,test_ids=test)
    assert [r['id'] for r in w.rows(out/'inputs.jsonl')]==[records[0]['id']]


def test_changed_prepared_inputs_rejected(tmp_path):
    source,meta,_=dataset(tmp_path)
    out=prep(tmp_path,source,meta)
    with (out/'inputs.jsonl').open('a') as f:f.write('\n')
    with pytest.raises(ValueError,match='Prepared file changed'):
        w.bundle(out)


@pytest.mark.parametrize('track,expected',[('realism',10),('archetype',3)])
def test_drame_plans_all_dimensions_without_loading_model(tmp_path,track,expected):
    source,meta,_=dataset(tmp_path,track)
    out=prep(tmp_path,source,meta,track)
    checkpoint=tmp_path/'adapter';checkpoint.mkdir()
    (checkpoint/'adapter_config.json').write_text('{}')
    output,plans=w.infer_plan(infer_args(out,backend='drame',checkpoint=checkpoint))
    assert len(plans)==expected
    assert all('--use_gt_thinking' in p['argv'] for p in plans)
    assert not output.exists()


def test_fewshot_rejects_overlap_even_when_training_bundle_prepared(tmp_path):
    source,meta,records=dataset(tmp_path)
    out=prep(tmp_path,source,meta)
    for row in records:row['split']='train'
    w.write_rows(meta,records)
    examples=prep(tmp_path,source,meta,purpose='train',output=tmp_path/'examples')
    with pytest.raises(ValueError,match='overlap'):
        w.infer_plan(infer_args(out,backend='gemini25',model='test-model',examples=examples))


def test_score_matches_prepared_ids_and_traces_inputs(tmp_path):
    source,meta,records=dataset(tmp_path)
    out=prep(tmp_path,source,meta)
    pred=source/'pred.jsonl'
    w.write_rows(pred,[{'id':r['id'],'response':dict.fromkeys(w.DIMENSIONS['realism'],i+1)} for i,r in enumerate(records)])
    output=w.score(argparse.Namespace(workdir=out,name=None,predictions=[pred],n_boot=0,seed=1,output=None))
    result=json.loads(output.read_text())
    assert result['overall']['mean_dimension_pearson']==pytest.approx(1)
    assert result['workflow']['manifest_sha256']==w.digest(out/'run.json')


def test_offline_demo_runs_as_documented(tmp_path):
    result=cli('demo','--output',tmp_path/'demo')
    assert result.returncode==0,result.stderr
    output=json.loads((tmp_path/'demo/prepared/scores.json').read_text())
    assert output['workflow']['demo_fixture'] is True
    assert 'not model judgments' in result.stdout


@pytest.mark.parametrize('section',['zeroshot','fewshot'])
def test_cloud_cli_help_has_no_local_model_dependency(section):
    result=subprocess.run([sys.executable,str(ROOT/section/'pyscripts/eval.py'),'--help'],
                          cwd=ROOT,capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    assert '--model_tag' in result.stdout


def test_peft_loader_attaches_adapter_and_uses_base_processor(tmp_path,monkeypatch):
    spec=importlib.util.spec_from_file_location('loader',ROOT/'finetuning/src/utils/model_loading.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    (tmp_path/'adapter_config.json').write_text(json.dumps({'base_model_name_or_path':'base-model'}))
    calls=[]
    def load_base(base,**kwargs):calls.append(('base',base));return 'base-instance'
    def load_adapter(base,path):calls.append(('adapter',base,path));return 'adapted-instance'
    monkeypatch.setitem(sys.modules,'torch',SimpleNamespace(bfloat16='bf16'))
    monkeypatch.setitem(sys.modules,'transformers',SimpleNamespace(
        AutoProcessor=SimpleNamespace(from_pretrained=lambda base:('processor',base)),
        Qwen2AudioForConditionalGeneration=SimpleNamespace(from_pretrained=load_base)))
    monkeypatch.setitem(sys.modules,'peft',SimpleNamespace(PeftModel=SimpleNamespace(from_pretrained=load_adapter)))
    model,processor=module.load_model_and_processor(str(tmp_path))
    assert model=='adapted-instance'
    assert processor==('processor','base-model')
    assert calls==[('base','base-model'),('adapter','base-instance',str(tmp_path))]


@pytest.mark.parametrize('section',['zeroshot','fewshot'])
def test_gemini_uses_callers_key_and_needs_no_private_project(section,monkeypatch):
    import os
    for key in ['GOOGLE_GENAI_USE_VERTEXAI','GOOGLE_CLOUD_PROJECT','GOOGLE_API_KEY','GEMINI_API_KEY']:
        monkeypatch.delenv(key,raising=False)
    monkeypatch.setenv('GEMINI_API_KEY','fixture-key')
    tree=ast.parse((ROOT/section/'models/gemini25_pro.py').read_text())
    init=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='__init__')
    ns={'os':os,'Optional':__import__('typing').Optional,'genai':SimpleNamespace(Client=lambda **kwargs:kwargs)}
    exec(compile(ast.Module(body=[init],type_ignores=[]),'gemini-init','exec'),ns)
    model=SimpleNamespace()
    ns['__init__'](model,'gemini25','fixture-model')
    assert model.client=={'vertexai':False,'api_key':'fixture-key'}


def test_process_launch_to_named_predictions_to_scoring(tmp_path):
    source,meta,records=dataset(tmp_path)
    work=prep(tmp_path,source,meta)
    output=work/'predictions'/'fixture-child'
    # An actual child process exercises the file handoff without an API/GPU.
    code=('import json,pathlib; inputs=pathlib.Path('+repr(str(work/'inputs.jsonl'))+'); '
          'out=pathlib.Path('+repr(str(output/'trial1.jsonl'))+'); '
          'rows=[json.loads(x) for x in inputs.read_text().splitlines()]; '
          'dims='+repr(w.DIMENSIONS['realism'])+'; '
          'out.write_text("".join(json.dumps({"id":r["id"],"response":dict.fromkeys(dims,i+1)})+"\\n" '
          'for i,r in enumerate(rows)))')
    w.execute_plan(output,[{'cwd':str(ROOT),'argv':[sys.executable,'-c',code]}],False)
    assert (output/'commands.json').is_file()
    scored=w.score(argparse.Namespace(workdir=work,name='fixture-child',predictions=None,n_boot=0,seed=1,output=None))
    assert json.loads(scored.read_text())['coverage']['n_usable_prediction_ids']==4


def test_declared_audio_checksum_and_prepared_audio_changes_are_checked(tmp_path):
    source,meta,records=dataset(tmp_path,count=1)
    records[0]['audio']['synthesized']['sha256']='bad-checksum'
    w.write_rows(meta,records)
    with pytest.raises(ValueError,match='checksum mismatch'):
        prep(tmp_path,source,meta)
    records[0]['audio']['synthesized']['sha256']=w.digest(source/'test_0.wav')
    w.write_rows(meta,records)
    out=prep(tmp_path,source,meta)
    (out/'audio/test_0.wav').write_bytes(b'corrupted')
    with pytest.raises(ValueError,match='Prepared file changed'):
        w.bundle(out,audio=True)
    # Scoring saved predictions does not require keeping a staged audio copy.
    assert w.bundle(out)['n_records']==1


def test_failed_audio_conversion_leaves_no_prepared_directory(tmp_path):
    source,meta,_=dataset(tmp_path,count=1)
    (source/'test_0.wav').write_bytes(b'not audio')
    with pytest.raises(Exception):
        prep(tmp_path,source,meta)
    assert not (tmp_path/'prepared').exists()


def test_downloaded_placeholder_is_not_presented_as_runnable_data(tmp_path,monkeypatch):
    monkeypatch.setattr(w,'require_packages',lambda names:None)
    monkeypatch.setitem(sys.modules,'huggingface_hub',SimpleNamespace(snapshot_download=lambda **kwargs:str(tmp_path)))
    (tmp_path/'README.md').write_text('Data payload pending')
    with pytest.raises(ValueError,match='no metadata payload'):
        w.fetch(argparse.Namespace(repo='ftshijt/speech-drame-evalbench',revision='fixture-revision',output=tmp_path))
