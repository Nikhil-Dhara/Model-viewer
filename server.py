import tornado.ioloop
import tornado.web
import tornado.locks
from tornado.options import define, options, parse_command_line
import csv, json, io
from run_application import Predictor
import tornado.httputil

define("debug", default=True, help="run in debug mode")


class ExampleResource():
    def __init__(self):
        super().__init__()
        self.cond = tornado.locks.Condition()
        self.counter = 0

    def hello(self, v):
        self.counter += v
        print(self.counter)
        return self.counter


class MainHandler(tornado.web.RequestHandler):

    def get(self):
        self.render("templates/index.html")

    async def post(self):
        arg_dic = {}
        file_dic = {}
        tornado.httputil.parse_body_arguments(self.request.headers["Content-Type"], self.request.body, arg_dic,
                                              file_dic)
        lines = io.StringIO(file_dic['file'][0]['body'].decode('utf-8'))
        target_column = arg_dic['target_column'][0].decode('utf-8')
        learning_rate = float(arg_dic['learning_rate'][0].decode('utf-8'))
        n_iterations = int(arg_dic['n_iterations'][0].decode('utf-8'))
        source_column = [i.decode('utf-8') for i in arg_dic['source_columns'][0].splitlines()]
        predictor = Predictor(lines, target_column, source_column)
        predictions, predicted_probabilities = predictor.run_model(learning_rate=learning_rate, n_iterations=n_iterations)
        accuracy = predictor.get_performance(predictions)
        res_dict, auc = predictor.roc(predicted_probabilities, predictions)
        await self.render("templates/result.html", auc=auc, accuracy=accuracy, series=json.dumps(res_dict).replace('"', ''))


def make_app():
    return tornado.web.Application([
        (r"/", MainHandler),
    ])


if __name__ == "__main__":
    app = make_app()
    app.listen(8888)
    tornado.ioloop.IOLoop.current().start()