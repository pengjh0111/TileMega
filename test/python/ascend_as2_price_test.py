"""CPU-only accounting checks; fixture costs are not hardware calibration."""
import unittest
import gzip
import hashlib
import json
from pathlib import Path
import tempfile
from tilemega.bench.as2_native_prices import task_overheads,tile_body_prices,cost_measurement
from tilemega.bench.as2_body_profile import (logical_bytes,profile_bytes,attention_intervals,
    ATTENTION_PHASE_MAGIC,ATTENTION_PHASE_NAMES)
from tilemega.bench.as2_models import RUNS


class ProfileArchive(unittest.TestCase):
    def test_attention_phase_ownership_extent_and_same_lane_order(self):
        legacy=[0]*26
        self.assertIsNone(attention_intervals(legacy,True,False,1))
        for participant,mask in [(False,0x50),(True,0xaf)]:
            row=[0]*26;row[3:5]=[100,1000];row[8:10]=[ATTENTION_PHASE_MAGIC,mask]
            for i in range(8):
                if mask&(1<<i):row[10+2*i:12+2*i]=[120+100*i,170+100*i]
            phases=attention_intervals(row,True,participant,2)
            self.assertEqual(phases,{n:100 for i,n in enumerate(ATTENTION_PHASE_NAMES) if mask&(1<<i)})
            with self.assertRaises(ValueError):attention_intervals(row,True,not participant,2)
            with self.assertRaises(ValueError):attention_intervals(row,False,participant,2)
            first=next(i for i in range(8) if mask&(1<<i))
            bad=row.copy();bad[11+2*first]=1001
            with self.assertRaises(ValueError):attention_intervals(bad,True,participant,2)
            bad=row.copy();bad[10+2*first]=99
            with self.assertRaises(ValueError):attention_intervals(bad,True,participant,2)
            if not participant:
                bad=row.copy();bad[22:24]=[160,170]
                with self.assertRaises(ValueError):attention_intervals(bad,True,participant,2)

    def test_lossless_archive_and_identity_rejection(self):
        with tempfile.TemporaryDirectory(dir=RUNS/'probes') as raw:
            source=Path(raw)/'profile.bin';archive=source.with_name(source.name+'.gz')
            data=bytes(range(256))*4;source.write_bytes(data)
            self.assertEqual(profile_bytes(source),data)
            archive.write_bytes(gzip.compress(data));receipt=source.with_name(source.name+'.archive.json')
            identity=dict(source=str(source),archive=str(archive),roundtrip_verified=True,bytes=len(data),
                source_sha256=hashlib.sha256(data).hexdigest(),archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
            receipt.write_text(json.dumps(identity));source.unlink()
            self.assertEqual(profile_bytes(source),data)
            receipt.write_text(json.dumps(dict(identity,source='/tmp/escape')))
            with self.assertRaisesRegex(ValueError,'identity'):profile_bytes(source)
            receipt.write_text(json.dumps(identity));archive.write_bytes(archive.read_bytes()+b'changed')
            with self.assertRaisesRegex(ValueError,'identity'):profile_bytes(source)


class AscendPriceAccounting(unittest.TestCase):
    def setUp(self):
        self.cost=dict(dispatcher_aic_ns=10.,dispatcher_aiv_ns=20.,cross_worker_ready_ns=3.)

    def stage(self,kind,variant=0,epilogue=0):
        fields=[0]*33;fields[17]=kind;fields[18]=epilogue;fields[31]=variant
        return fields

    def test_calibration_partitions_roles_at_the_measured_group_count(self):
        job=dict(executable_sha256='fixture-not-calibration',timing_status='contended',
            physical_device_id=1,started_at='a',finished_at='b',pid=1,start_ticks=2)
        for groups in (8,14,28):
            value=dict(status='passed',device_abort=0,groups=groups,workers=3*groups,
                payload_samples_passed=1024*groups,repeats=32,
                dispatcher_ticks_sum=[64]*groups+[128]*(2*groups),aiv_copy_ticks=[256]*(2*groups),
                roundtrip_ticks_sum=640,copy_logical_read_write_bytes=4096)
            row=cost_measurement(value,job)
            self.assertEqual((row['groups'],row['dispatcher_aic_ns'],row['dispatcher_aiv_ns'],
                              row['cross_worker_ready_ns'],row['gm_bytes_per_ns']),(groups,2.,4.,10.,16.))
            for key,replacement in [('workers',3*groups-1),('payload_samples_passed',1),
                                    ('aiv_copy_ticks',[256]),('repeats',0)]:
                with self.assertRaises(ValueError):cost_measurement(dict(value,**{key:replacement}),job)

    def test_owner_and_participant_service_spans(self):
        for variant in (4,5,6,7,8):
            self.assertEqual(task_overheads(self.stage(13,variant),dict(internal_mix_coordination_included=True),self.cost),(10.,0.))
        self.assertEqual(task_overheads(self.stage(0,1,1),{},self.cost),(30.,6.))
        self.assertEqual(task_overheads(self.stage(0),{},self.cost),(10.,0.))
        self.assertEqual(task_overheads(self.stage(13,3),{},self.cost),(20.,0.))
        self.assertEqual(task_overheads(self.stage(1,3),{},self.cost),(20.,0.))

    def test_wrong_span_rejected_before_pricing(self):
        for stage,flag in [(self.stage(13,6),False),(self.stage(13,3),True),(self.stage(0,1,1),True)]:
            with self.assertRaises(ValueError):task_overheads(stage,dict(internal_mix_coordination_included=flag),self.cost)

    def test_valid_qk_traffic_preserves_physical_tile_tail_and_full_pv(self):
        base=self.stage(13,6);base[23:26]=[8,128,2];base[26]=1088
        base[5]=base[6]=(1<<64)-1
        valid=base.copy();valid[31]=8;m=[0]*15;m[9]=1
        self.assertEqual(logical_bytes(base,m,0,64)-logical_bytes(valid,m,0,64),514560)
        self.assertEqual(logical_bytes(base,m,0,1086),logical_bytes(valid,m,0,1086))
        # Crossing64 keys adds exactly one original tile, rather than shortening
        # the last MMAD or changing its stride/reduction tree.
        self.assertEqual(logical_bytes(valid,m,0,64)-logical_bytes(valid,m,0,63),
                         (logical_bytes(base,m,0,64)-logical_bytes(base,m,0,63))+4*(2*128+64*128+2*64))

    def test_split_and_combine_share_one_physical_traffic_budget(self):
        partial=self.stage(0,1);partial[20:26]=[64,2048,6144,3,2048,1]
        combine=partial.copy();combine[17]=6;combine[18]=1
        self.assertEqual(logical_bytes(partial,[],0,0),598016)
        self.assertEqual(logical_bytes(partial,[],191,0),598016)
        self.assertEqual(logical_bytes(combine,[],0,0),40960)
        partial[20]=combine[20]=1
        self.assertEqual(logical_bytes(partial,[],47,0),528896)
        self.assertEqual(logical_bytes(combine,[],15,0),2560)
        self.assertEqual(task_overheads(combine,{},self.cost),(20.,0.))
        with self.assertRaises(ValueError):logical_bytes(partial,[],48,0)
        partial[24]=1024
        with self.assertRaises(ValueError):logical_bytes(partial,[],0,0)
        partial[20]=64;partial[22]=8192;partial[24]=2731
        self.assertEqual(logical_bytes(partial,[],0,0),794720)
        self.assertEqual(logical_bytes(partial,[],191,0),794432)
        partial[21]=combine[21]=3072;combine[22]=8192;partial[23]=combine[23]=2
        partial[24]=combine[24]=4096;combine[20]=64
        self.assertEqual(logical_bytes(partial,[],191,0),1187840)
        self.assertEqual(logical_bytes(combine,[],95,0),32768)

    def test_separate_key_partials_store_once_and_pack_once(self):
        partial=self.stage(0,1);partial[20:26]=[64,1024,4096,7,586,1]
        combine=partial.copy();combine[17]=6
        self.assertEqual(logical_bytes(partial,[],0,0),176960)
        self.assertEqual(logical_bytes(partial,[],223,0),175232)
        self.assertEqual(logical_bytes(combine,[],31,0),61440)
        wrong=partial.copy();wrong[21]=2048
        with self.assertRaises(ValueError):logical_bytes(wrong,[],0,0)
        combine[18]=1
        with self.assertRaises(ValueError):logical_bytes(combine,[],0,0)
        pack=self.stage(25);pack[20]=64;pack[23:28]=[8,768,1,128,4]
        self.assertEqual(logical_bytes(pack,[],63,0),24576)
        with self.assertRaises(ValueError):logical_bytes(pack,[],64,0)

    def test_local_tail_partials_are_private_body_traffic(self):
        local=self.stage(0,2,1);local[20:24]=[1024,2048,8192,0]
        base=local.copy();base[31]=1
        for k in (6144,8192):
            local[22]=base[22]=k
            self.assertEqual(logical_bytes(local,[],13,0),logical_bytes(base,[],13,0))
            self.assertEqual(logical_bytes(local,[],14,0)-logical_bytes(base,[],14,0),56*16*128)
        local[22]=7168
        with self.assertRaises(ValueError):logical_bytes(local,[],14,0)
        local[22]=6144
        local[20]=64
        with self.assertRaises(ValueError):logical_bytes(local,[],14,0)

    def test_coordinate_prices_use_observed_regions_without_zero_fallback(self):
        stage=self.stage(0,2,1);stage[19:23]=[1024,1024,2048,8192]
        fixture=dict(body_service_regions=dict(regular=dict(tasks=896,median=10.),tail=dict(tasks=128,median=70.)))
        for k in (6144,8192):
            stage[22]=k
            result=tile_body_prices(stage,fixture)
            self.assertEqual([result[i] for i in (13,14,15,16)],[10.,70.,70.,10.])
            self.assertEqual(result.count(70.),128)
        stage[22]=7168
        with self.assertRaises(ValueError):tile_body_prices(stage,fixture)
        stage[22]=6144
        fixture['body_service_regions']['tail']['median']=0
        with self.assertRaises(ValueError):tile_body_prices(stage,fixture)

    def test_q_tail_traffic_and_coordinate_price_domain(self):
        stage=self.stage(0,3,4);stage[19:24]=[2048,1024,4096,4096,0]
        base=stage.copy();base[31]=1
        self.assertEqual(logical_bytes(stage,[],27,0),logical_bytes(base,[],27,0))
        self.assertEqual(logical_bytes(stage,[],28,0)-logical_bytes(base,[],28,0),24*16*128)
        fixture=dict(body_service_regions=dict(regular=dict(tasks=1792,median=10.),tail=dict(tasks=256,median=30.)))
        values=tile_body_prices(stage,fixture)
        self.assertEqual(values[27:33],[10.,30.,30.,30.,30.,10.])
        self.assertEqual(values.count(30.),256)
        for k in (4096,14336):
            residual=stage.copy();residual[18]=1;residual[22]=k
            ordinary=residual.copy();ordinary[31]=1
            self.assertEqual(logical_bytes(residual,[],28,0)-logical_bytes(ordinary,[],28,0),24*16*128)
            self.assertEqual(tile_body_prices(residual,fixture).count(30.),256)
        for index,value in [(20,16),(21,2048),(22,8192),(18,0)]:
            invalid=stage.copy();invalid[index]=value
            with self.assertRaises(ValueError):tile_body_prices(invalid,fixture)
        fixture['body_service_regions']['tail']['median']=0
        with self.assertRaises(ValueError):tile_body_prices(stage,fixture)


if __name__=='__main__':unittest.main()
