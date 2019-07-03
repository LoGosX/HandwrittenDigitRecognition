fetch('/hello')
    .then(function (response) {
        return response.text();
    }).then(function (text) {
        console.log('GET response text: ');
        console.log(text);
    });

fetch('/hello')
    .then(function (response) {
        return response.json();
    })
    .then(function (json) {
        console.log('GET response as JSON: ');
        console.log(json);
    });