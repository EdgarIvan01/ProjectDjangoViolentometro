$(document).ready(function() {
    // Código personalizado con jQuery
    /*$('#guardar').click(function() {
        alert('¡Botón clickeado!');
    });*/

    $('#datos').submit(function(event) {
        event.preventDefault(); // Evita que se envíe el formulario de forma tradicional
        texto_1 = $.trim($("#input1").val());
        texto_2 = $.trim($("#input2").val());
        var url = "../secuencial/"+texto_1+"&"+texto_2

        console.log(texto_1+" "+texto_2+" "+url)
        //location.href = url;

        /*$.ajax({
            type: 'POST',
            url: url,
            dataType: "json",
            contentType: 'json',
            data: {texto_1:texto_1, texto_2:texto_2},
            success: function(data) {
                console.log(data); // Maneja la respuesta del servidor
            },
            error: function(xhr, errmsg, err) {
                //console.log(xhr.status + ': ' + xhr.responseText); // Maneja errores si los hay
            }
        });*/
    });
});