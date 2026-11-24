"""Observable contracts for direct and composed detection evaluation."""
import unittest
from unittest.mock import patch
from klygo import metrics, outputs


def frame(boxes=(), identity="image", index=0, size=(200,200)):
    return outputs.Detection([outputs.Box(coords, label=label, score=score)
                              for coords,label,score in boxes],
                             source_id=identity,frame_index=index,image_size=size)


class TestMetrics(unittest.TestCase):
    def setUp(self):
        self.target = frame([((0,0,10,10),"cat",1),((20,20,30,30),"dog",1)])
        self.pred = frame([((0,0,10,10),"cat",.9),((0,0,10,10),"cat",.8),
                           ((20,20,30,30),"cat",.7),((50,50,60,60),"dog",.6)])

    def test_counts_and_matches(self):
        result=metrics.precision_recall(self.pred,self.target)
        self.assertEqual((result['tp'],result['fp'],result['fn']),(1,3,1))
        self.assertEqual(result['precision'],.25)
        self.assertEqual(result['recall'],.5)
        self.assertAlmostEqual(result['f1'],1/3)
        self.assertEqual(result['mean_iou'],1)
        matched=metrics.match_boxes(self.pred,self.target)[0]
        self.assertEqual(matched['matches'],[dict(pred_index=0,target_index=0,iou=1)])
        self.assertEqual(matched['unmatched_pred'],[1,2,3])
        self.assertEqual(matched['unmatched_target'],[1])

    def test_every_metric_direct_and_grouped_for_both_types(self):
        names=[n for n in metrics.__all__ if n not in ('compose','evaluate')]
        configurations=[getattr(metrics,n)() for n in names]
        evaluator=metrics.compose(*configurations)
        before=self.pred.to_dict(),self.target.to_dict()
        for pred,target in ((self.pred,self.target),
                            (outputs.Detections([self.pred]),outputs.Detections([self.target]))):
            result=metrics.evaluate(pred,target,evaluator)
            for name in names:
                self.assertEqual(result[name],getattr(metrics,name)(pred,target),name)
        self.assertEqual(before,(self.pred.to_dict(),self.target.to_dict()))

    def test_ap_ranking_and_absent_classes(self):
        target=frame([((0,0,10,10),'cat',1)])
        pred=frame([((40,40,50,50),'cat',.9),((0,0,10,10),'cat',.8),
                    ((60,60,70,70),'ghost',.7)])
        result=metrics.map_score(pred,target)
        self.assertAlmostEqual(result['map'],.5)
        self.assertEqual(result['map_50'],result['map_75'])
        self.assertIsNone(result['per_class'][1]['ap'])
        missed=metrics.map_score(frame(),target)
        self.assertEqual(missed['map'],0)

    def test_ap_101_point_recall(self):
        pred=frame([((0,0,10,10),'cat',1)])
        target=frame([((0,0,10,10),'cat',1),((20,20,30,30),'cat',1)])
        self.assertAlmostEqual(metrics.map_score(pred,target)['map'],51/101)

    def test_empty_and_prediction_only(self):
        empty=frame()
        self.assertIsNone(metrics.map_score(empty,empty)['map'])
        self.assertIsNone(metrics.calibration(empty,empty)['ece'])
        self.assertEqual(metrics.precision_recall(empty,empty)['f1'],0)
        self.assertEqual(metrics.precision_recall(self.pred,empty)['fp'],4)
        self.assertEqual(metrics.precision_recall(empty,self.target)['fn'],2)
        result=metrics.evaluate(outputs.Detections(),outputs.Detections())
        self.assertIsNone(result['map_score']['map'])

    def test_identity_alignment_and_validation(self):
        a=frame([((0,0,10,10),'cat',1)],identity='a')
        b=frame(identity='b')
        result=metrics.per_image(outputs.Detections([a,b]),outputs.Detections([b,a]))
        self.assertEqual([r['identity'] for r in result],[a.identity,b.identity])
        self.assertEqual(result[0]['tp'],1)
        for pred,target in ((a,b),(a,frame(identity='a',size=(100,100)))):
            with self.assertRaises(ValueError):
                metrics.evaluate(pred,target)
        with self.assertRaises(TypeError):
            metrics.evaluate(a,outputs.Detections([a]))
        with self.assertRaises(TypeError):
            metrics.precision_recall(pred=a)
        with self.assertRaises(TypeError):
            metrics.evaluate([],[])

    def test_configuration_validation(self):
        for operation in (lambda:metrics.compose(),
                          lambda:metrics.compose(metrics.per_image(),metrics.per_image()),
                          lambda:metrics.calibration(bins=0),lambda:metrics.calibration(bins=True),
                          lambda:metrics.error_analysis(iou=.1,bg_iou=.1),
                          lambda:metrics.iou_sweep(thresholds=[]),
                          lambda:metrics.iou_sweep(thresholds=[.5,.5]),
                          lambda:metrics.precision_recall(iou=float('nan')),
                          lambda:metrics.precision_recall(iou=True)):
            with self.assertRaises(ValueError): operation()
        with self.assertRaises(TypeError): metrics.compose('per_image')
        with self.assertRaises(TypeError): metrics.calibration(bins=1.5)
        with self.assertRaises(TypeError): metrics.evaluate(self.pred,self.target,metrics.per_image())

    def test_confusion_and_error_partition(self):
        result=metrics.confusion_matrix(self.pred,self.target)
        self.assertEqual(result['labels'],['cat','dog',None])
        self.assertEqual(result['matrix'],[[1,0,0],[1,0,0],[1,1,0]])
        counts=metrics.error_analysis(self.pred,self.target)['counts']
        self.assertEqual(counts,dict(duplicate=1,classification=1,localization=0,
                                   classification_localization=0,background=1,missed=1))
        pred=frame([((5,0,15,10),'cat',.9),((25,20,35,30),'cat',.8)])
        counts=metrics.error_analysis(pred,self.target)['counts']
        self.assertEqual(counts['localization'],1)
        self.assertEqual(counts['classification_localization'],1)
        self.assertEqual(sum(v for k,v in counts.items() if k!='missed'),2)

    def test_calibration_bins_and_endpoints(self):
        pred=frame([((0,0,10,10),'cat',1),((50,50,60,60),'cat',0)])
        result=metrics.calibration(pred,self.target,bins=2)
        self.assertEqual(result['ece'],0)
        self.assertEqual([r['count'] for r in result['bins']],[1,1])
        result=metrics.calibration(self.pred,self.target,bins=2)
        self.assertAlmostEqual(result['ece'],.5)

    def test_sweep_and_stricter_iou(self):
        rows=metrics.threshold_sweep(self.pred,self.target,thresholds=[1,.8,0])
        self.assertEqual([r['threshold'] for r in rows],[0,.8,1])
        self.assertEqual([(r['tp'],r['fp'],r['fn']) for r in rows],[(1,3,1),(1,1,1),(0,0,2)])
        pred=frame([((0,0,9,10),'cat',.9)])
        rows=metrics.iou_sweep(pred,self.target,thresholds=[.95,.5])
        self.assertEqual([r['tp'] for r in rows],[1,0])

    def test_size_boundaries_and_gt_assignment(self):
        target=frame([((0,0,31,31),'cat',1),((0,0,32,32),'cat',1),((0,0,96,96),'cat',1)])
        rows=metrics.size_breakdown(target,target)
        self.assertEqual([r['support'] for r in rows],[1,1,1])
        self.assertEqual([r['ap'] for r in rows],[1,1,1])
        target=frame([((0,0,32,32),'cat',1)])
        pred=frame([((0,0,31,31),'cat',.9)])
        rows=metrics.size_breakdown(pred,target)
        self.assertEqual(rows[0]['tp'],0)
        self.assertEqual(rows[1]['tp'],1)

    def test_class_ids_do_not_define_labels(self):
        pred=outputs.Detection([outputs.Box((0,0,10,10),label='cat',class_id=99)],
                               source_id='image',image_size=(200,200))
        self.assertEqual(metrics.precision_recall(pred,self.target)['tp'],1)

    def test_default_evaluation(self):
        self.assertEqual(set(metrics.evaluate(self.pred,self.target)),{'precision_recall','map_score'})

    def test_shared_iou_and_immutable_threshold_configuration(self):
        from klygo.metrics import _core
        thresholds = [.5]
        evaluator = metrics.compose(metrics.precision_recall(), metrics.per_class(),
                                    metrics.map_score(thresholds=thresholds))
        thresholds.append(.9)
        with patch.object(_core, '_iou_matrix', wraps=_core._iou_matrix) as overlap:
            result = metrics.evaluate(self.pred, self.target, evaluator)
            self.assertEqual(overlap.call_count, 1)
        self.assertEqual(len(result['map_score']['per_iou']), 1)

    def test_stream_is_rejected_without_consumption(self):
        consumed = []
        def frames():
            consumed.append(True)
            yield self.pred
        stream = outputs.Detections(frames(), stream=True)
        with self.assertRaises(ValueError):
            metrics.evaluate(stream, outputs.Detections([self.target]))
        self.assertEqual(consumed, [])
        stream.close()

    def test_packed_storage_and_global_ap_ranking(self):
        pred = outputs.Detections([
            frame([((0,0,10,10),'cat',.8)], identity='a'),
            frame([((40,40,50,50),'cat',.9)], identity='b'),
        ], packed=True)
        target = outputs.Detections([
            frame([((0,0,10,10),'cat',1)], identity='a'), frame(identity='b'),
        ], packed=True)
        self.assertAlmostEqual(metrics.map_score(pred, target)['map'], .5)
        self.assertTrue(pred.is_packed)

    def test_size_ap_preserves_confidence_tie_input_order(self):
        pred = frame([((40,40,50,50),'cat',.9), ((0,0,10,10),'cat',.9)])
        target = frame([((0,0,10,10),'cat',1)])
        self.assertAlmostEqual(metrics.size_breakdown(pred,target)[0]['ap'], .5)

    def test_reports_can_be_saved_as_json(self):
        import tempfile
        from klygo import files
        with tempfile.TemporaryDirectory() as directory:
            path = files.path(directory) / 'report.json'
            evaluator = metrics.compose(*(getattr(metrics,n)() for n in metrics.__all__
                                          if n not in ('compose','evaluate')))
            report = metrics.evaluate(self.pred,self.target,evaluator)
            files.save(path,report,overwrite=True,verbose=False)
            saved = files.load(path,verbose=False)
            self.assertEqual(saved['precision_recall'],report['precision_recall'])

    def test_f1_score_direct_and_detection_evaluation(self):
        direct = metrics.f1_score(precision=0.8, recall=0.5)
        self.assertAlmostEqual(direct['f1'], 2 * 0.8 * 0.5 / (0.8 + 0.5))
        self.assertEqual((direct['precision'], direct['recall']), (0.8, 0.5))

        zero = metrics.f1_score(precision=0.0, recall=0.0)
        self.assertEqual(zero['f1'], 0.0)

        pooled = metrics.precision_recall(self.pred, self.target)
        det_f1 = metrics.f1_score(self.pred, self.target)
        self.assertAlmostEqual(det_f1['f1'], pooled['f1'])
        self.assertAlmostEqual(det_f1['precision'], pooled['precision'])
        self.assertAlmostEqual(det_f1['recall'], pooled['recall'])

        composed = metrics.evaluate(self.pred, self.target, metrics.compose(metrics.f1_score()))
        self.assertAlmostEqual(composed['f1_score']['f1'], pooled['f1'])

        for bad in (
            lambda: metrics.f1_score(precision=0.5),
            lambda: metrics.f1_score(recall=0.5),
            lambda: metrics.f1_score(precision=1.2, recall=0.5),
            lambda: metrics.f1_score(precision=True, recall=0.5),
            lambda: metrics.f1_score(self.pred, self.target, precision=0.5, recall=0.5),
        ):
            with self.assertRaises(ValueError):
                bad()


if __name__=='__main__':
    unittest.main()
