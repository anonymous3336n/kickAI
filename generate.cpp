#include <openssl/ec.h>
#include <openssl/pem.h>
#include <openssl/err.h>
#include <iostream>
#include <stdexcept>

void generate_ec_key(const std::string& private_key_file, const std::string& public_key_file) {
    EC_KEY *ec_key = nullptr;
    BIO *pri_bio = nullptr;
    BIO *pub_bio = nullptr;

    try {
        // Create a new EC key pair
        ec_key = EC_KEY_new_by_curve_name(NID_X9_62_prime256v1);
        if (!ec_key) {
            throw std::runtime_error("Failed to create EC Key structure");
        }

        // Generate a key pair
        if (EC_KEY_generate_key(ec_key) != 1) {
            throw std::runtime_error("Failed to generate EC key");
        }

        // Write the private key to the file
        pri_bio = BIO_new_file(private_key_file.c_str(), "w");
        if (!pri_bio) {
            throw std::runtime_error("Failed to open private key file");
        }
        if (PEM_write_bio_ECPrivateKey(pri_bio, ec_key, nullptr, nullptr, 0, nullptr, nullptr) != 1) {
            throw std::runtime_error("Failed to write private key to file");
        }

        // Write the public key to the file
        pub_bio = BIO_new_file(public_key_file.c_str(), "w");
        if (!pub_bio) {
            throw std::runtime_error("Failed to open public key file");
        }
        if (PEM_write_bio_EC_PUBKEY(pub_bio, ec_key) != 1) {
            throw std::runtime_error("Failed to write public key to file");
        }

        std::cout << "EC key pair generated successfully." << std::endl;
    } catch (const std::runtime_error& e) {
        std::cerr << e.what() << std::endl;
        if (pri_bio) BIO_free_all(pri_bio);
        if (pub_bio) BIO_free_all(pub_bio);
        if (ec_key) EC_KEY_free(ec_key);
        return;
    }

    // Free up resources
    if (pri_bio) BIO_free_all(pri_bio);
    if (pub_bio) BIO_free_all(pub_bio);
    if (ec_key) EC_KEY_free(ec_key);
}

int main() {
    const std::string private_key_file = "private_key.pem";
    const std::string public_key_file = "public_key.pem";

    generate_ec_key(private_key_file, public_key_file);

    return 0;
}