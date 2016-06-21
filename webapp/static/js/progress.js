var source = new EventSource("/progress");
source.addEventListener('message', function(e) {
    var data = JSON.parse(e.data);
    $('#status').text(data.message);
    $('.progress-bar').css('width', data.progress+'%').attr('aria-valuenow', data.progress);

    if(data.message == 'Done!') {
        window.location.href = data.back;
    }
}, false);