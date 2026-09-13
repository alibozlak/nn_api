# nn_api

An HTTP/JSON API over [neuralflow](https://crates.io/crates/neuralflow). JSON goes in,
JSON comes back, so a client in any language -- Java here -- can build, train and
query a neural network without linking Rust. It runs beside that client, on
the same machine.

The endpoints follow Keras, because neuralflow does:

| Keras | this API |
| --- | --- |
| `Sequential([Input(shape=(2,)), Dense(8, activation='relu')])` + `compile(...)` | `POST /create-compile-model` |
| `model.fit(X, y, epochs=500)` | `POST /models/{id}/train` |
| `model.predict(X)` | `POST /models/{id}/predict` |
| `model.evaluate(X, y)` | `POST /models/{id}/evaluate` |
| `model.get_weights()` / `set_weights(...)` | `GET` / `PUT /models/{id}/weights` |
| `model.summary()` | `GET /models/{id}` (the `summary` field) |
| `model.export("m.onnx", format="onnx")` | `GET /models/{id}/onnx` |

## Running it

```bash
cargo run --release          # debug builds train roughly 10x slower
curl http://127.0.0.1:8079/health
```

| Variable | Default | What it does |
| --- | --- | --- |
| `NN_API_ADDR` | `127.0.0.1:8079` | Address to bind. It has to be a loopback address; anything else is refused at startup. The default is deliberately not 8080, Spring Boot's own, so a Spring Boot client on the same machine does not collide with it. |
| `NN_API_ALLOW_REMOTE` | unset | `yes` lifts that check, for when something that authenticates sits in front of the server. |
| `NN_API_DATA_DIR` | unset | Directory `x_path` and `y_path` are read from. Unset, a request naming a path is refused. |
| `NN_API_MAX_FILE_MB` | `256` | Biggest file `x_path` or `y_path` may point at. |
| `NN_API_BODY_LIMIT_MB` | `32` | Biggest request body, so the cap on data sent inline. Data read through `x_path` and `y_path` is capped by `NN_API_MAX_FILE_MB` instead. |
| `NN_API_LOG` | `info` | `tracing` filter, e.g. `debug,tower_http=debug`. |
| `NN_API_CORS` | unset | `permissive` turns CORS on. Browsers only -- a Java client never sends a preflight. |
| `NN_API_MAX_MODELS` | `256` | Models held at once. |
| `NN_API_MAX_EPOCHS` | `100000` | Epochs one train call may ask for. |
| `NN_API_MAX_SAMPLES` | `1000000` | Rows of `x` one call may carry. |
| `NN_API_MAX_LAYERS` / `NN_API_MAX_UNITS` / `NN_API_MAX_FEATURES` | `64` / `4096` / `4096` | Model size. |

## The endpoints

Every request that has a body needs `Content-Type: application/json`, and every
response is JSON -- the ONNX export is the one exception.

```text
GET    /health                  is the server up, and how many models it holds
POST   /create-compile-model    create and compile a model
GET    /models                  list them, newest first
GET    /models/{id}             one model, with Keras' summary table
DELETE /models/{id}             204, and the id stops working
POST   /models/{id}/train       fit
POST   /models/{id}/predict     predict
POST   /models/{id}/evaluate    evaluate, without training
GET    /models/{id}/weights     get_weights
PUT    /models/{id}/weights     set_weights
GET    /models/{id}/onnx        the model as an ONNX file (bytes, not JSON)
POST   /utils/scale             column_based_scaling
```

`/models` itself only lists; creating one is `POST /create-compile-model`,
named after the two Keras steps it does in one call.

Some of these paths are things and some are calls -- `train`, `predict` and
`scale` are verbs, not resources -- which is why this reads HTTP/JSON rather
than REST. Every request is still self-contained: it carries the model id, and
the server remembers nothing about the client between calls.

### `POST /create-compile-model`

```json
{
  "name": "xor",
  "features": 2,
  "seed": 1234,
  "layers": [
    { "units": 8, "activation": "relu",    "name": "layer1" },
    { "units": 1, "activation": "sigmoid", "name": "layer2" }
  ],
  "loss": "binary_crossentropy",
  "optimizer": { "type": "adam", "learning_rate": 0.05 },
  "batch_size": 4
}
```

- `features` -- Keras' `Input(shape=(features,))`: columns per sample.
- `activation` -- `relu`, `sigmoid` or `linear`.
- `loss` -- `binary_crossentropy` (or `bce`) or `mean_squared_error` (or `mse`).
  It can be left out here and sent with the training data instead.
- `optimizer` -- `"adam"`, `"sgd"`, or the object above. A bare name means the
  optimizer's own default rate: 0.001 for Adam, 0.01 for SGD. Left out entirely
  while a loss is set, it is Adam at 0.001, and the response says so.
- `name` on a layer is optional. Without one the layers are named the way Keras
  names them: `dense`, `dense_1`, `dense_2`. The name is what `PUT /weights`
  matches on, so it is worth setting.
- `seed` fixes the initial weights and the batch order. **Leaving it out, or
  sending `null`, is not the same as having no seed**: the server draws one
  from the system's randomness and reports it in the response, so the model
  can be built again exactly by sending that number back. Every model has a
  seed, and `GET /models/{id}` always says which.
- `batch_size` is the samples per gradient step this model trains with, kept
  so a train request need not repeat it. Keras' own 32 when left out. Keras has
  no model-level `batch_size` -- it is an argument to `fit` there -- and a train
  request can still override it for a single run.
- `weights` (optional) loads known weights into the new model, which is how a
  client carries a trained model over a restart. See below.

Answers `201` with the model, Keras' summary table included:

```json
{
  "id": "0583b9da-ba2a-4737-88ca-410819418c85",
  "name": "xor",
  "features": 2,
  "output_units": 1,
  "total_params": 33,
  "layers": [
    { "name": "layer1", "units": 8, "activation": "relu", "input_count": 2, "params": 24 },
    { "name": "layer2", "units": 1, "activation": "sigmoid", "input_count": 8, "params": 9 }
  ],
  "loss": "binary_crossentropy",
  "optimizer": { "type": "adam", "learning_rate": 0.05 },
  "seed": 1234,
  "batch_size": 4,
  "trained_epochs": 0,
  "last_loss": null,
  "created_at_ms": 1757752440123,
  "updated_at_ms": 1757752440123,
  "summary": "Model: \"sequential\"\n____..."
}
```

### `POST /models/{id}/train`

```json
{
  "x": [[0, 0], [0, 1], [1, 0], [1, 1]],
  "y": [[0], [1], [1], [0]],
  "epochs": 500,
  "batch_size": 32,
  "shuffle": true,
  "return_history": false
}
```

One row of `x` is one sample, and the same row of `y` is its target. `y` needs
one column per unit of the last layer. `epochs` defaults to 1, as in Keras, and
`batch_size` to whatever the model was created with; a batch bigger than the
data set means one step per epoch. Either may be sent here, and the response
says which values the run actually used.

`loss` and `optimizer` may be sent here too, which compiles the model
the way calling `compile` again would, and they stay set afterwards. A `seed`
sent here applies to this run only and does not replace the model's own; left
out, the model's seed is used, which is what makes a run repeatable.

Because the seed is set again at the start of every run, two runs of the same
length over the same data shuffle in the same order -- that is what repeatable
means here. Send a different `seed` with each call if you would rather they
differ.

`return_history: false` leaves the per-epoch losses out, which matters when the
run is tens of thousands of epochs long.

```json
{
  "id": "0583b9da-...",
  "epochs": 500,
  "batch_size": 32,
  "samples": 4,
  "loss_function": "binary_crossentropy",
  "optimizer": { "type": "adam", "learning_rate": 0.05 },
  "initial_loss": 0.752320560664955,
  "final_loss": 0.0011514305974509322,
  "loss": [0.752, 0.694, ...],
  "trained_epochs": 500,
  "duration_ms": 10
}
```

Training is cumulative: a second train call carries on from the weights the
first one left, and `trained_epochs` counts every epoch the model has ever run.
Adam's momentum, though, starts fresh on each call -- one long call and two
short ones are not quite the same run.

### `POST /models/{id}/predict`

```json
{ "x": [[0, 0], [0, 1], [1, 0], [1, 1]] }
```

```json
{
  "id": "0583b9da-...",
  "rows": 4,
  "columns": 1,
  "predictions": [[0.0038], [0.9997], [0.9997], [0.0001]]
}
```

### `POST /models/{id}/evaluate`

`{ "x": [...], "y": [...] }`, optionally with a `loss` to measure against
instead of the compiled one. It answers `{ "loss": 0.0011, "samples": 4, ... }`
and changes nothing about the model.

### Data from a file instead of the body

`train`, `predict` and `evaluate` can read `x` and `y` from JSON files rather
than the request body, which matters once a data set outgrows a request:

```json
{ "x_path": "sets/train_x.json", "y_path": "sets/train_y.json", "epochs": 500 }
```

Each file holds exactly what the field would have held -- `[[0, 0], [0, 1], ...]`.
Send `x` or `x_path`, not both; the two ways can be mixed, `x` from a file and
`y` inline.

The server has to be started with `NN_API_DATA_DIR`, and every path is taken
relative to that directory:

```bash
NN_API_DATA_DIR=/home/me/nn_data cargo run --release
```

Letting a caller choose a file on the server's disk is how arbitrary files get
read, so the directory is a hard boundary: absolute paths and `..` are refused,
and the path is canonicalised so a symbolic link pointing out of the directory
is refused too. Being bound to `127.0.0.1` does not make this unnecessary --
every user on the machine can reach the server, not only your application.

### `GET` and `PUT /models/{id}/weights`

Keras' `get_weights` and `set_weights`. `W` is `input_count` rows of `units`
values, `b` is one value per unit:

```json
{
  "layers": [
    { "name": "layer2", "weights": [[1.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]], "bias": [0.5] }
  ]
}
```

A `PUT` may name only some layers; the rest keep what they have. Models live in
memory only, so this pair is also how a client persists one: `GET` the weights,
store them, and send them back as `weights` in a later
`POST /create-compile-model`.

### `GET /models/{id}/onnx`

The trained model as an [ONNX](https://onnx.ai) file -- the only endpoint that
answers with bytes (`application/octet-stream`) rather than JSON. Every runtime
reads it: `onnxruntime` in Python or C++, `tract` in Rust, ONNX Runtime for
Java. The weights are written as f32, so predictions move by about `1e-7`.

### `POST /utils/scale`

`column_based_scaling`: every column is divided by a power of ten taken from
**that column's first row**. `y` must have exactly one column.

```json
{ "x": [[1500, 3], [2500, 4]], "y": [[250000], [300000]] }
```

```json
{ "x": [[1.5, 3.0], [2.5, 4.0]], "y": [[2.5], [3.0]], "ten_power_ratios": [3, 0, 5] }
```

The last entry of `ten_power_ratios` belongs to `y`: multiply a prediction by
`10^5` to read it in the original scale.

## Errors

Every failure -- a bad path, unparseable JSON, a shape mismatch, a panic inside
neuralflow -- comes back in one shape, so the client can parse errors the same
way it parses results:

```json
{ "error": { "code": "invalid_request", "message": "the model takes 2 features per sample, but row 0 of 'x' has 3" } }
```

| Status | `code` | When |
| --- | --- | --- |
| 400 | `invalid_request` | The values do not fit the model: a wrong feature count, `epochs: 0`, an id that is not a UUID. |
| 400 | `invalid_json` | The body is not JSON. |
| 404 | `not_found` | No model with that id, or no such path. |
| 405 | `method_not_allowed` | The path exists, but not with this method. |
| 409 | `conflict` | `NN_API_MAX_MODELS` reached. |
| 413 | `payload_too_large` | The body is bigger than `NN_API_BODY_LIMIT_MB`. Send the data through `x_path` and `y_path` instead. |
| 415 | `unsupported_media_type` | `Content-Type: application/json` is missing. |
| 422 | `invalid_json` | The JSON parsed but does not fit the request type -- an unknown field, `"activation": "softmax"`. The message names the field and lists what is allowed. |
| 422 | `engine_error` | neuralflow refused the values. Its own message is passed through. |
| 500 | `internal_error` | A bug here. |

neuralflow reports every bad argument by panicking. The validation in front of
it turns the ones it knows about into a 400 that says what to fix, and every
call into the engine is wrapped, so a panic becomes a 422 rather than a dropped
connection.

## Calling it from Java

JDK 11 and up, no dependencies beyond a JSON library:

```java
import java.net.URI;
import java.net.http.*;

HttpClient client = HttpClient.newHttpClient();
String base = "http://127.0.0.1:8079";

// 1. Create the model.
HttpResponse<String> created = client.send(
    HttpRequest.newBuilder(URI.create(base + "/create-compile-model"))
        .header("Content-Type", "application/json")
        .POST(HttpRequest.BodyPublishers.ofString("""
            {
              "name": "xor",
              "features": 2,
              "seed": 1234,
              "layers": [
                { "units": 8, "activation": "relu",    "name": "layer1" },
                { "units": 1, "activation": "sigmoid", "name": "layer2" }
              ],
              "loss": "binary_crossentropy",
              "optimizer": { "type": "adam", "learning_rate": 0.05 }
            }"""))
        .build(),
    HttpResponse.BodyHandlers.ofString());

String id = new ObjectMapper().readTree(created.body()).get("id").asText();

// 2. Train it.
client.send(
    HttpRequest.newBuilder(URI.create(base + "/models/" + id + "/train"))
        .header("Content-Type", "application/json")
        .POST(HttpRequest.BodyPublishers.ofString("""
            { "x": [[0,0],[0,1],[1,0],[1,1]],
              "y": [[0],[1],[1],[0]],
              "epochs": 500, "return_history": false }"""))
        .build(),
    HttpResponse.BodyHandlers.ofString());

// 3. Ask it something.
HttpResponse<String> prediction = client.send(
    HttpRequest.newBuilder(URI.create(base + "/models/" + id + "/predict"))
        .header("Content-Type", "application/json")
        .POST(HttpRequest.BodyPublishers.ofString("""
            { "x": [[0,1],[1,1]] }"""))
        .build(),
    HttpResponse.BodyHandlers.ofString());
```

Three things to watch on the Java side:

- **`Content-Type: application/json` is required.** Without it the server
  answers 415, and `HttpClient` does not add it for you.
- **The wire format is `snake_case`.** Mapping it onto Java records without an
  annotation on every field takes one line of Jackson setup:

  ```java
  ObjectMapper mapper = new ObjectMapper()
      .setPropertyNamingStrategy(PropertyNamingStrategies.SNAKE_CASE);

  record Prediction(String id, int rows, int columns, double[][] predictions) {}
  record TrainResult(int epochs, int samples, double initialLoss, double finalLoss,
                     int trainedEpochs, long durationMs) {}
  ```

- **Read the status before the body.** Every non-2xx answer is the error
  envelope above, never the type you asked for.

## What it does not do

- **Models live in memory.** A restart loses them. `GET /weights` and the
  `weights` field of `POST /create-compile-model` are how a client keeps one;
  nothing is written to disk here. It also means one process owns the models:
  two instances behind a load balancer would not see each other's.
- **Localhost only, and no authentication.** The server refuses to start on
  an address other machines could reach. Everyone *on* the machine can still
  call it. `NN_API_ALLOW_REMOTE=yes` lifts the check, which is only sensible
  with something that authenticates in front of it.
- **One training run per model at a time.** A second train call on the same
  model waits for the first; runs on *different* models go in parallel. A long
  run blocks nothing else, though: predictions on the model being trained are
  answered from the weights that run started with, and training happens on a
  blocking thread pool, so no endpoint stalls behind it.
- **What neuralflow has.** `Dense` layers with `relu`, `sigmoid` or `linear`,
  two losses, two optimizers. No softmax, no convolution, no recurrence.

## Development

```bash
cargo test      # 26 integration tests over the real router, 2 unit tests
cargo clippy --all-targets
```

`tests/api.rs` drives the same router the binary serves, without a socket.
