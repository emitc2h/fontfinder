var source = new EventSource("/progress");
source.onmessage = function(event) {
    $('.progress-bar').css('width', event.data+'%').attr('aria-valuenow', event.data);   
}