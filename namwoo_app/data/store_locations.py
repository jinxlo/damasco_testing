STORES_LIST = [
    {
        "city": "Caracas",
        "whsName": "Almacen Principal SABANA GRANDE",
        "address": "",
    },
    {"city": "Caracas", "whsName": "Almacen Principal CCCT", "address": ""},
    {"city": "Caracas", "whsName": "Damasco La Trinidad", "address": ""},
    {
        "city": "Valencia",
        "whsName": "Tienda VALENCIA CENTRO",
        "address": "Avenida 103 Carabobo, Valencia (Diagonal al Gran Bazar centro). 0412-3715859. Horario 9:00 am - 6:00 pm",
    },
    {
        "city": "Valencia",
        "whsName": "Almacen SAN DIEGO",
        "address": "Av Rojas Queipo, entre Av. Bolívar y distribuidor Paseo Cabriales. (Antiguo concesionario de carros). 04125136702",
    },
    {
        "city": "Maracay",
        "whsName": "Tienda MARACAY LAS DELICIAS",
        "address": "Av. Bolívar, Al lado del Aero Expresos y de la estación de servicio San Jacinto, Maracay, Aragua. 0412-6452756. Horario 9:00 am - 6:00 pm",
    },
    {
        "city": "Cagua",
        "whsName": "Tienda CAGUA",
        "address": "Carretera Nacional Cagua La Villa, entre calle Las Acacias y Diego de Lozada, Cagua, Aragua. 0412-6119852. Horario 9:00 am - 6:00 pm",
    },
    {
        "city": "Barquisimeto",
        "whsName": "Tienda BARQUISIMETO CENTRO",
        "address": "Av. Venezuela entre calles 10 y 11, Barquisimeto, Lara. 0412-3167538. Horario 9:00 am - 6:00 pm",
    },
    {
        "city": "Maracaibo",
        "whsName": "Tienda MARACAIBO BELLA VISTA",
        "address": "Circunvalación 1, Av. 33 con calle 93, antiguo supermercado NASA, Maracaibo, Zulia. 0412-6192424. Horario 9:00 am - 6:00 pm",
    },
    {
        "city": "Guatire",
        "whsName": "Tienda GUATIRE BUENAVENTURA",
        "address": "Avenida Intercomunal, frente al C.C. Buena Aventura Place, Guatire, Miranda. 0412-3167516. Horario 9:00 am - 6:00 pm",
    },
    {
        "city": "Lechería",
        "whsName": "Tienda LECHERIA C.C. Caribbean",
        "address": "Av. Intercomunal Jorge Rodríguez, Edif Yaveth, sector Vistamar, Lechería, Anzoátegui. 0412-6192419. Horario 9:00 am - 6:00 pm",
    },
    {
        "city": "Maturín",
        "whsName": "Tienda MATURIN CENTRO",
        "address": "Av. Alirio Ugarte Pelayo, Maturín 6201, Monagas",
    },
    {
        "city": "Puerto Ordaz",
        "whsName": "Tienda PUERTO ORDAZ ALTA VISTA",
        "address": "Urb. los Samanes, sector AltaVista, frente al centro medico Orinokia, Puerto Ordaz, Bolívar. 0412-3715863. Horario 9:00 am - 6:00 pm",
    },
    {
        "city": "Trujillo",
        "whsName": "Tienda TRUJILLO CENTRO",
        "address": "Urbanización La plata, frente la redoma Las banderas al lado del banco Provincial, Valera, Trujillo. Horario 9:00 am - 6:00 pm",
    },
    {
        "city": "San Cristóbal",
        "whsName": "Tienda SAN CRISTOBAL BARRIO OBRERO",
        "address": "Calle 14 con carrera 4 al lado de la iglesia de la Ermita, sector centro, San Cristobal, Táchira. 0412-3167522. Horario 9:00 am - 6:00 pm",
    },
    {"city": "San Felipe", "whsName": "Tienda SAN FELIPE CENTRO", "address": ""},
    {
        "city": "Puerto La Cruz",
        "whsName": "Tienda PUERTO LA CRUZ PLAZA MAYOR",
        "address": "Calle Sucre con calle Libertad a pocos metros de la plaza Bolivar, Puerto La Cruz, Anzoátegui. Horario 9:00 am - 6:00 pm",
    },
    {"city": "La Guaira", "whsName": "Tienda LA GUAIRA TERMINAL", "address": ""},
]

CITY_TO_WAREHOUSES = {}
for store in STORES_LIST:
    CITY_TO_WAREHOUSES.setdefault(store["city"], []).append(store["whsName"])

VALID_CITIES = sorted(CITY_TO_WAREHOUSES.keys())
