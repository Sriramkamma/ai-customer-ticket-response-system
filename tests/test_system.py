import json
import tempfile
import threading
import unittest
from unittest.mock import patch
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.request import Request, urlopen

import server

class SystemTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory(); server.DB_PATH=Path(cls.temp.name)/'test.db'
        cls.http=ThreadingHTTPServer(('127.0.0.1',0),server.Handler)
        cls.thread=threading.Thread(target=cls.http.serve_forever,daemon=True); cls.thread.start()
        cls.base=f'http://127.0.0.1:{cls.http.server_port}'
    @classmethod
    def tearDownClass(cls):
        cls.http.shutdown(); cls.http.server_close(); cls.temp.cleanup()
    def api(self,path,method='GET',data=None):
        req=Request(self.base+path,data=json.dumps(data).encode() if data is not None else None,method=method,headers={'Content-Type':'application/json'})
        try:
            with urlopen(req) as r: return r.status,json.loads(r.read())
        except Exception as e:
            if hasattr(e,'code'): return e.code,json.loads(e.read())
            raise
    def test_predicts_distinct_ticket_types(self):
        model=server.Predictor()
        self.assertIsNotNone(model.classifier)
        self.assertIsNotNone(model.vectorizer.vocabulary_)
        self.assertIn('password',model.predict('Unable to login. Forgot password and account access'))
        self.assertIn('tracking',model.predict('Where is my order? Shipment delivery tracking is late'))
        self.assertIn('billing',model.predict('My card payment was declined at checkout'))
        self.assertIn(model.predict('My card payment was declined at checkout'),model.classifier.classes_)
    def test_predictor_handles_empty_unknown_and_insufficient_data(self):
        model=server.Predictor()
        fallback='Review the issue details and follow the relevant support procedure.'
        self.assertEqual(model.predict(''),fallback)
        self.assertEqual(model.predict('qzxv blorp unknownword'),fallback)
        with tempfile.TemporaryDirectory() as directory:
            dataset=Path(directory)/'training.json'
            dataset.write_text(json.dumps([
                {'subject':'Only example','description':'single class','solution':'One answer.'}
            ]),encoding='utf-8')
            with patch.object(server,'DATASET',dataset):
                insufficient=server.Predictor()
        self.assertIsNone(insufficient.classifier)
        self.assertTrue(insufficient.training_warning)
        self.assertEqual(insufficient.predict('Only example'),fallback)
    def test_local_retrieval_and_grounded_response(self):
        docs=server.retrieve('password reset login account')
        self.assertTrue(docs); self.assertEqual(docs[0]['source'],'accounts.md')
        with patch.object(server,'LLM_URL','http://127.0.0.1:1/unavailable'):
            result,engine,sources=server.grounded_response({'subject':'login','description':'password issue'},'Reset password.',docs)
        self.assertIn('couldn\'t generate a detailed response',result)
        self.assertNotIn('Reset password',result)
        self.assertTrue(sources)
        self.assertEqual(engine,'ollama_unavailable_fallback')
    def test_no_knowledge_and_retrieval_failure_have_distinct_safe_fallbacks(self):
        ticket={'subject':'Unknown issue','description':'No matching help article'}
        no_match, no_match_engine, no_match_sources = server.grounded_response(
            ticket, 'Unsupported predicted action.', []
        )
        retrieval_error, error_engine, error_sources = server.grounded_response(
            ticket, 'Unsupported predicted action.', [], retrieval_failed=True
        )
        self.assertIn('couldn\'t find enough information',no_match)
        self.assertIn('contact our support team',no_match.lower())
        self.assertNotIn('Unsupported predicted action',no_match)
        self.assertEqual(no_match_engine,'no_knowledge_fallback')
        self.assertEqual(no_match_sources,[])
        self.assertIn('couldn\'t access',retrieval_error)
        self.assertEqual(error_engine,'retrieval_unavailable_fallback')
        self.assertEqual(error_sources,[])
    def test_ticket_retrieval_query_includes_prediction_and_ticket_details(self):
        ticket={'subject':'App crashes','description':'It closes when opened'}
        solution='Restart the app and device.'
        query=server.build_retrieval_query(ticket,solution)
        self.assertEqual(query,'App crashes It closes when opened Restart the app and device.')

        with patch.object(server.Predictor,'predict',return_value=solution), \
             patch.object(server,'retrieve_with_status',return_value=([],None)) as retrieval:
            create_data={
                'customer_name':'Example Customer',
                'subject':ticket['subject'],
                'description':ticket['description'],
            }
            connection=server.db()
            required_field=connection.execute(
                'SELECT 1 FROM field_definitions WHERE name = ?',
                ('test_required',),
            ).fetchone()
            connection.close()
            if required_field:
                create_data['dynamic_values']={'test_required':'CASE-987'}
            status,created=self.api('/api/tickets','POST',create_data)
        self.assertEqual(status,201)
        retrieval.assert_called_once_with(query)
        self.assertEqual(created['ai_engine'],'no_knowledge_fallback')
    def test_ticket_questions_use_shared_rag_and_pass_context_to_ollama(self):
        questions = [
            ('The application keeps crashing when I open it. What should I do.', 'Restart the app and device, then install the latest available app update.', 'app_and_files.md'),
            ('I forgot my password and cannot log in.', 'Reset the password using the account recovery process.', 'accounts.md'),
            ('I was charged twice for my order. Can I get a refund?', 'Share the order details with support so the duplicate charge can be reviewed.', 'billing.md'),
            ('My order is delayed and has not been delivered. How can I track it?', 'Check the order tracking page for the latest shipment status.', 'orders.md'),
        ]
        for question, solution, expected_source in questions:
            with self.subTest(source=expected_source):
                query = server.build_retrieval_query(
                    {'subject':question,'description':''},solution
                )
                docs = server.retrieve(query)
                self.assertTrue(docs)
                self.assertEqual(docs[0]['source'], expected_source)

        app_question, app_solution, _ = questions[0]
        app_docs = server.retrieve(server.build_retrieval_query(
            {'subject':app_question,'description':''},app_solution
        ))
        captured = {}

        class OllamaReply:
            def __init__(self, generated):
                self.generated=generated
            def __enter__(self):
                return self
            def __exit__(self, *_):
                return False
            def read(self):
                return json.dumps({'response': self.generated}).encode()

        def fake_ollama(request, timeout):
            captured.update(json.loads(request.data))
            generated=(
                'Restart the app and device, then check for an available app update.'
                if 'check for an available app update' in captured['prompt'].lower()
                else 'Please contact support for help.'
            )
            return OllamaReply(generated)

        with patch.object(server.urllib.request, 'urlopen', side_effect=fake_ollama):
            response, engine, sources = server.grounded_response(
                {'subject': app_question, 'description': ''},
                app_solution,
                app_docs,
            )

        self.assertEqual(engine, 'ollama')
        self.assertEqual(sources, ['app_and_files.md'])
        self.assertIn('Contact support if the issue continues', captured['prompt'])
        self.assertIn('app_and_files.md', captured['prompt'])
        self.assertIn(app_question, captured['prompt'])
        self.assertIn(app_solution, captured['prompt'])
        self.assertIn('check for an available app update', response.lower())

    def test_retrieval_model_failure_is_reported_separately(self):
        with patch.object(server,'_embedding_model',None), \
             patch.object(server,'_faiss_index',None), \
             patch.object(server,'_semantic_documents',[]), \
             patch.object(server,'_semantic_signature',None), \
             patch.object(server,'SentenceTransformer',side_effect=RuntimeError('model unavailable')):
            documents,error=server.retrieve_with_status('An example issue')
        self.assertEqual(documents,[])
        self.assertIsInstance(error,RuntimeError)
    def test_order_delivery_response_excludes_unrelated_knowledge(self):
        query='Where is my order? My shipment delivery is late and tracking has not updated.'
        docs=server.retrieve(query)
        self.assertTrue(docs)
        self.assertEqual(docs[0]['source'],'orders.md')
        combined=' '.join(d['content'] for d in docs).lower()
        self.assertNotIn('password',combined)
        self.assertNotIn('payment',combined)
        self.assertNotIn('billing',combined)
        with patch.object(server,'LLM_URL','http://127.0.0.1:1/unavailable'):
            result,engine,sources=server.grounded_response({'subject':'Where is my order?','description':'My shipment delivery is late and tracking has not updated.'},'Check the order tracking page for the latest shipment status.',docs)
        self.assertEqual(engine,'ollama_unavailable_fallback')
        self.assertIn('contact our support team',result.lower())
        self.assertNotIn('tracking page',result.lower())
        self.assertNotIn('password',result.lower())
        self.assertNotIn('payment',result.lower())
        self.assertEqual(sources,['orders.md'])
    def test_ticket_api_persists_delivery_response_without_unrelated_knowledge(self):
        payload={
            'customer_name':'Example Customer',
            'subject':'my order delivery got delayed',
            'description':'my order was scheduled for 15 October, but I have not received an update.',
        }
        with patch.object(server,'LLM_URL','http://127.0.0.1:1/unavailable'):
            status,created=self.api('/api/tickets','POST',payload)
            self.assertEqual(status,201)
            status,saved=self.api(f"/api/tickets/{created['id']}")
        self.assertEqual(status,200)
        self.assertEqual(saved['response'],created['response'])
        response=saved['response'].lower()
        self.assertIn('couldn\'t generate a detailed response',response)
        self.assertNotIn('shipping address',response)
        self.assertNotIn('password',response)
        self.assertNotIn('reset link',response)
        self.assertNotIn('payment',response)
        self.assertNotIn('refund',response)
        self.assertNotIn('for a missing package',response)
        self.assertEqual(created['ai_engine'],'ollama_unavailable_fallback')
    def test_frontend_assets_are_served(self):
        for path,expected in [('/','text/html'),('/app.js','text/javascript'),('/styles.css','text/css')]:
            with urlopen(self.base+path) as response:
                self.assertEqual(response.status,200); self.assertIn(expected,response.headers.get('Content-Type',''))
    def test_ticket_create_update_read_soft_delete_and_dynamic_field(self):
        status,_=self.api('/api/fields'); self.assertEqual(status,200)
        status,_=self.api('/api/fields','POST',{'name':'test_required','label':'Case reference','type':'text','required':True})
        self.assertEqual(status,201)
        payload={'customer_name':'Example Customer','subject':'Unable to log in','description':'Forgot password and cannot sign in','dynamic_values':{'test_required':'CASE-123'}}
        status,ticket=self.api('/api/tickets','POST',payload); self.assertEqual(status,201)
        self.assertTrue(ticket['predicted_solution']); self.assertTrue(ticket['response']); ident=ticket['id']
        status,read=self.api(f'/api/tickets/{ident}'); self.assertEqual(read['dynamic_values']['test_required'],'CASE-123')
        status,updated=self.api(f'/api/tickets/{ident}','PUT',{'response':'Edited response','status':'Resolved'}); self.assertEqual(updated['response'],'Edited response')
        status,result=self.api(f'/api/tickets/{ident}','DELETE'); self.assertTrue(result['deleted'])
        status,_=self.api(f'/api/tickets/{ident}'); self.assertEqual(status,404)
    def test_dynamic_field_types_validation_and_ticket_editing(self):
        status,_=self.api('/api/fields','POST',{
            'name':'refund_amount',
            'label':'Refund amount',
            'type':'number',
        })
        self.assertEqual(status,201)
        values={
            'order_number':'ORDER-500',
            'contact_preference':'Email',
            'purchase_date':'2026-10-28',
            'refund_amount':'24.50',
            'urgent_callback':False,
        }
        payload={
            'customer_name':'Dynamic Field Customer',
            'subject':'Question with all dynamic fields',
            'description':'Please save each field type.',
            'dynamic_values':values,
        }
        status,created=self.api('/api/tickets','POST',payload)
        self.assertEqual(status,201)
        ident=created['id']
        status,read=self.api(f'/api/tickets/{ident}')
        self.assertEqual(status,200)
        self.assertEqual(read['dynamic_values'],values)

        updated_values={**values,'order_number':'ORDER-501','urgent_callback':True}
        status,updated=self.api(f'/api/tickets/{ident}','PUT',{
            'dynamic_values':updated_values,
            'status':'In Progress',
        })
        self.assertEqual(status,200)
        self.assertEqual(updated['dynamic_values'],updated_values)
        self.assertEqual(updated['status'],'In Progress')

        for bad_values in (
            {**values,'purchase_date':'not-a-date'},
            {**values,'purchase_date':'2026-02-30'},
            {**values,'refund_amount':True},
            {**values,'refund_amount':'NaN'},
            {**values,'contact_preference':'Text message'},
            {**values,'urgent_callback':'false'},
        ):
            with self.subTest(bad_values=bad_values):
                status,_=self.api('/api/tickets','POST',{
                    **payload,
                    'dynamic_values':bad_values,
                })
                self.assertEqual(status,400)

        status,_=self.api('/api/fields','POST',{
            'name':'required_test_checkbox',
            'label':'Required confirmation',
            'type':'checkbox',
            'required':True,
        })
        self.assertEqual(status,201)
        required_ticket={**payload,'dynamic_values':{**values,'required_test_checkbox':False}}
        status,_=self.api('/api/tickets','POST',required_ticket)
        self.assertEqual(status,400)
        required_ticket['dynamic_values']['required_test_checkbox']=True
        status,_=self.api('/api/tickets','POST',required_ticket)
        self.assertEqual(status,201)
        connection=server.db()
        connection.execute('DELETE FROM field_definitions WHERE name = ?',('required_test_checkbox',))
        connection.commit()
        connection.close()

    def test_dynamic_field_validation_rejects_invalid_shapes_and_missing_required(self):
        fields=[
            {'name':'text','label':'Text','type':'text','required':True,'options':[]},
            {'name':'dropdown','label':'Dropdown','type':'dropdown','required':True,'options':['One','Two']},
            {'name':'number','label':'Number','type':'number','required':True,'options':[]},
            {'name':'date','label':'Date','type':'date','required':True,'options':[]},
            {'name':'checkbox','label':'Checkbox','type':'checkbox','required':True,'options':[]},
        ]
        valid={'text':'hello','dropdown':'One','number':'2.5','date':'2026-10-28','checkbox':True}
        self.assertEqual(server.validate_dynamic(valid,fields),[])
        self.assertTrue(server.validate_dynamic({},fields))
        self.assertTrue(server.validate_dynamic({'text':'ok','dropdown':'Invalid','number':'bad','date':'2026-02-30','checkbox':False},fields))
        self.assertTrue(server.validate_dynamic([],fields))

    def test_dynamic_field_api_rejects_invalid_definitions(self):
        for data in (
            {'name':'empty_options','label':'Empty choices','type':'dropdown','options':[]},
            {'name':'bad_option','label':'Bad option','type':'dropdown','options':['']},
            {'name':'bad_required','label':'Bad required','type':'text','required':'yes'},
        ):
            with self.subTest(data=data):
                status,_=self.api('/api/fields','POST',data)
                self.assertEqual(status,400)
    def test_validation_and_missing_ticket_data(self):
        status,_=self.api('/api/tickets','POST',{'customer_name':'','subject':'','description':''}); self.assertEqual(status,400)
        status,_=self.api('/api/tickets/99999'); self.assertEqual(status,404)

if __name__=='__main__': unittest.main()
